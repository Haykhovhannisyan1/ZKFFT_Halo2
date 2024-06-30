use halo2_proofs::pasta::Fp;
use halo2_proofs::poly::Rotation;
use halo2_proofs::{circuit::*, plonk::*};
use ff::Field;
use halo2_proofs::plonk::{
    Advice, Column, ConstraintSystem, Error,
};

#[derive(Debug, Clone)]
pub struct SelectConfig {
    pub advice: [Column<Advice>; 3],
    pub selector: [Selector; 2],
}
pub struct SelectChip {
    pub config: SelectConfig,
}

impl SelectChip {
    pub fn construct(config: SelectConfig) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<Fp>, advice: Vec<Column<Advice>>) -> SelectConfig {
        let col_a = advice[0];
        let col_b = advice[1];
        let col_c = advice[2];
        let selector_1 = meta.selector();
        let selector_2 = meta.selector();
        let constant = meta.fixed_column();

        meta.enable_equality(col_c);
        meta.enable_constant(constant);

        meta.create_gate("subtract", |meta| {
            let s = meta.query_selector(selector_1);
            let a = meta.query_advice(col_a, Rotation::cur());
            let b = meta.query_advice(col_b, Rotation::cur());
            let c = meta.query_advice(col_c, Rotation::cur());
            vec![s * (a - b - c)]
        });

        meta.create_gate("running product", |meta| {
            let s = meta.query_selector(selector_2);
            let a = meta.query_advice(col_a, Rotation::cur());
            let b = meta.query_advice(col_b, Rotation::cur());
            let c = meta.query_advice(col_c, Rotation::cur());
            let c_prev = meta.query_advice(col_c, Rotation::prev());
            let sub = a - b;
            vec![s * (c_prev * sub - c)]
        });

        meta.create_gate("constant column", |meta| {
            let s = meta.query_selector(selector_2);
            let b = meta.query_advice(col_b, Rotation::cur());
            let b_prev = meta.query_advice(col_b, Rotation::prev());
            vec![s * (b_prev - b)]
        });

        SelectConfig {
            advice: ([col_a, col_b, col_c]),
            selector: ([selector_1, selector_2]),
        }
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fp>,
        commit: &Vec<Value<Fp>>,
        witness: &Value<Fp>,
        num_rows: usize,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        layouter.assign_region(
            || "select",
            |mut region| {
                let mut c_cell: AssignedCell<Fp, Fp>;

                self.config.selector[0].enable(&mut region, 0)?;

                region.assign_advice(|| "a", self.config.advice[0], 0, || commit[0])?;

                region.assign_advice(|| "b", self.config.advice[1], 0, || *witness)?;

                let c_val = commit[0].and_then(|commit| witness.map(|witness| commit - witness));
                c_cell = region.assign_advice(|| "a-b", self.config.advice[2], 0, || c_val)?;

                for row in 1..num_rows {
                    self.config.selector[1].enable(&mut region, row)?;

                    region.assign_advice(|| "a", self.config.advice[0], row, || commit[row])?;

                    region.assign_advice(|| "b", self.config.advice[1], row, || *witness)?;

                    let sub = commit[row].and_then(|c| witness.map(|w| c - w));

                    let c_val = c_cell.value().and_then(|d| sub.map(|c| *d * c));

                    c_cell =
                        region.assign_advice(|| "product", self.config.advice[2], row, || c_val)?;
                }

                region.constrain_constant(c_cell.cell(), Fp::ZERO)?;

                return Ok(c_cell);
            },
        )
    }
}
