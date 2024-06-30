use ff::Field;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::pasta::{pallas, Fp};
use halo2_proofs::plonk::{
    self, Advice, Column, Error,
};
use halo2_proofs::poly::Rotation;
use halo2_proofs::{circuit::*, plonk::*};
use pasta_curves::group::Curve;



/// This represents an advice column at a certain row in the ConstraintSystem
#[derive(Debug, Clone)]
pub struct PrmsConfig {
    pub advice: [Column<Advice>; 3],
    pub selector: Selector,
}
pub struct PrmsChip {
    pub config: PrmsConfig,
}

impl PrmsChip {
    pub fn construct(config: PrmsConfig) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut plonk::ConstraintSystem<Fp>,
        advice: Vec<Column<Advice>>,
    ) -> PrmsConfig {
        let col_a = advice[0];
        let col_b = advice[1];
        let col_c = advice[2];
        let selector_1 = meta.selector();

        meta.create_gate("is permissible", |meta| {
            //add alpha beta random numbers
            let s = meta.query_selector(selector_1);
            let c = meta.query_advice(col_c, Rotation::cur());
            let b = meta.query_advice(col_b, Rotation::cur());
            vec![s * (c.clone() * c.clone() - b)]
        });

        meta.create_gate("is point on curve", |meta| {
            let s = meta.query_selector(selector_1);
            let a = meta.query_advice(col_a, Rotation::cur());
            let b = meta.query_advice(col_b, Rotation::cur());

            let on_curve =
                b.square() - a.clone().square() * a - Expression::Constant(pallas::Affine::b());

            vec![s * on_curve]
        });

        PrmsConfig {
            advice: ([col_a, col_b, col_c]),
            selector: (selector_1),
        }
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<Fp>,
        x: &Value<Fp>,
        y: &Value<Fp>,
        y_sqrt: &Value<Fp>,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        layouter.assign_region(
            || "permissible",
            |mut region| {
                self.config.selector.enable(&mut region, 0)?;

                region.assign_advice(|| "a", self.config.advice[0], 0, || *x)?;

                region.assign_advice(|| "b", self.config.advice[1], 0, || *y)?;

                let c_cell =
                    region.assign_advice(|| "sqrt(y)", self.config.advice[2], 0, || *y_sqrt)?;

                return Ok(c_cell);
            },
        )
    }
}


pub fn permissible_commitment(
    c: &pallas::Point,
    h: &pallas::Point,
) -> (pallas::Point, u64) {
    let mut r = 0u64;
    let mut c_prime = *c;
    while !is_permissible(c_prime) {
        c_prime = (c_prime + h).into();
        r += 1;
    }
    (c_prime, r)
}

pub fn is_permissible(point: pallas::Point) -> bool {
    let y = point.to_affine().coordinates().expect("Couldn't get coordinates of a point");
    let y = y.y();
    if y.sqrt().is_none().into() {
        return  false;
    }
    else {
        return true;
    }
}