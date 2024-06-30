use ff::Field;
use halo2_gadgets::ecc::chip::FixedPoint;
use halo2_gadgets::utilities::lookup_range_check::LookupRangeCheckConfig;
use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::pasta::{pallas, EqAffine, Fp, Fq};
use halo2_proofs::plonk::{
    self, create_proof, keygen_pk, keygen_vk, verify_proof, BatchVerifier,
    Circuit, ConstraintSystem, Error, SingleVerifier,
};
use halo2_gadgets::ecc::{
        chip::{EccChip, EccConfig},
        ScalarFixed      
};
use halo2_gadgets::ecc::{FixedPoint as FPoint, Point};
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use halo2_proofs::{circuit::*, plonk::*};
use pasta_curves::arithmetic::CurveExt;
use pasta_curves::group::Group;
use pasta_curves::{Ep, EpAffine};
use rand_core::OsRng;
use rand::Rng;
use std::time::Instant;
use pasta_curves::group::Curve;

mod permissible;
mod select;
mod generator;

use permissible::*;
use select::*;
use generator::*;


/// This represents an advice column at a certain row in the ConstraintSystem
#[derive(Debug, Clone)]
struct MyConfig {
    pub select: SelectConfig,
    pub permisable: PrmsConfig,
    pub ecc_config: EccConfig<TestFixedBases>,
    pub instance: Column<Instance>,
}

#[derive(Default, Clone, Debug)]
struct MyCircuit {
    pub commits_x: Vec<Value<Fp>>,
    pub commits_y: Vec<Value<Fp>>,
    pub witness: Value<Fp>,
    pub w_sqrt: Value<Fp>,
    pub rerand_scalar: Value<Fq>,
    pub k: usize,
    pub index: usize,
}

impl Circuit<Fp> for MyCircuit {
    type Config = MyConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        MyChip::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let config_clone = config.clone();
        let chip = MyChip::construct(config_clone.select, config_clone.permisable, config.clone());
        chip.assign_select(
            layouter.namespace(|| "select"),
            &self.commits_x,
            &self.witness,
            self.k,
        );

        chip.assign_perm(
            layouter.namespace(|| "permissible"),
            &self.commits_x,
            &self.commits_y,
            &self.w_sqrt,
            self.index,
        );

        let ecc_chip = EccChip::construct(config.ecc_config.clone());
        chip.assign_rerand(layouter, ecc_chip.clone(), self.rerand_scalar, self.commits_x[self.index], self.commits_y[self.index]);

        Ok(())
    }
}

struct MyChip {
    select: SelectChip,
    permisable: PrmsChip,
    config: MyConfig
}

impl MyChip {
    fn construct(s_config: SelectConfig, p_config: PrmsConfig, config: MyConfig) -> Self {
        Self {
            select: SelectChip::construct(s_config),
            permisable: PrmsChip::construct(p_config),
            config,
        }
    }

    fn configure(meta: &mut plonk::ConstraintSystem<Fp>) -> MyConfig {
        let col_a = meta.advice_column();
        let col_b = meta.advice_column();
        let col_c = meta.advice_column();
        let col_d = meta.advice_column();
        let select_config = SelectChip::configure(meta, vec![col_a, col_b, col_c]);
        let prms_config = PrmsChip::configure(meta, vec![col_a, col_b, col_d]);

        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        for advice in advices.iter() {
            meta.enable_equality(*advice);
        }

        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];

        meta.enable_constant(lagrange_coeffs[0]);

        let table_idx = meta.lookup_table_column();
        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);

        let ecc_config =
        EccChip::<TestFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        MyConfig {
            select: select_config,
            permisable: prms_config,
            ecc_config,
            instance
        }
    }

    pub fn assign_select(
        &self,
        layouter: impl Layouter<Fp>,
        x: &Vec<Value<Fp>>,
        witness: &Value<Fp>,
        num_rows: usize,
    ) {
        SelectChip::assign(&self.select, layouter, x, witness, num_rows).expect("Select assignment Error");
    }

    //Soundness issue: reassigning row
    pub fn assign_perm( 
        &self,
        layouter: impl Layouter<Fp>,
        x: &Vec<Value<Fp>>,
        y: &Vec<Value<Fp>>,
        y_sqrt: &Value<Fp>,
        index: usize,
    ) {
        PrmsChip::assign(&self.permisable, layouter, &x[index], &y[index], y_sqrt).expect("Permisiible assignment Error");
    }

    pub fn assign_rerand(
        &self,
        mut layouter: impl Layouter<Fp>,
        ecc_chip: EccChip<TestFixedBases>,
        rerand_scalar: Value<Fq>,
        x_child: Value<Fp>,
        y_child: Value<Fp>,
    )
    {   
        let scalar = ScalarFixed::new(
            ecc_chip.clone(),
            layouter.namespace(|| "scalar"),
            rerand_scalar,
        ).expect("Couldn't witness scalar");

        let value_commit_r = ValueCommitR;
        let value_commit_r = FPoint::from_inner(ecc_chip.clone(), value_commit_r);
        let res = value_commit_r.mul(
            layouter.namespace(|| "rerand_scalar mul"), 
            scalar
        ).expect("Multiplication failed");

        let mut tmp_x = Fp::zero();
        x_child.map(|v| tmp_x = v);
        let mut tmp_y = Fp::zero();
        y_child.map(|v| tmp_y = v);
        let tmp = EpAffine::from_xy(tmp_x, tmp_y).expect("Couldn't construct point from x,y");

        let pt = Point::new(
            ecc_chip, 
            layouter.namespace(|| "witness non identity point"), 
            Value::known(tmp),
        ).expect("Couldn't witness a Point");

        let rerand = res.0.add(
            layouter.namespace(|| "rerandomized commitment"), 
            &pt).expect("Couldn't perform addition");
        
        layouter.constrain_instance(rerand.inner().x().cell(), self.config.instance, 0).expect("Couldn't constrain instance");
        layouter.constrain_instance(rerand.inner().y().cell(), self.config.instance, 1).expect("Couldn't constrain instance");

    }
}

fn keygen(k: u32, empty_circuit: MyCircuit) -> (Params<EqAffine>, ProvingKey<EqAffine>) {
    let params: Params<EqAffine> = Params::new(k);
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");
    (params, pk)
}

fn prover(
    params: &Params<EqAffine>,
    pk: &ProvingKey<EqAffine>,
    circuit: MyCircuit,
    public_input: &[&[Fp]],
) -> Vec<u8> {
    let rng = OsRng;
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof(params, pk, &[circuit], &[public_input], rng, &mut transcript)
        .expect("proof generation should not fail");
    transcript.finalize()
}

fn verifier(
    params: &Params<EqAffine>, 
    vk: &VerifyingKey<EqAffine>, 
    proof: &[u8], 
    public_input: &[&[Fp]],
    ) {
    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);
    assert!(verify_proof(params, vk, strategy, &[public_input], &mut transcript).is_ok());
}

fn main() {

    let k = 7;
    let depth = 6;
    println!("k = {}", k.clone()-1);
    println!("depth = {depth}");
    let index = 2;

    let iterations = 1 << k - 1;

    let hasher = pallas::Point::hash_to_curve("CT_COMMITMENT");
    let mut commitments: Vec<pallas::Point> = Vec::with_capacity(iterations);

    for _i in 0..iterations {
        let mut my_array: [u8; 11] = [0; 11];

        let mut rng = rand::thread_rng();
        for i in 0..11 {
            my_array[i] = rng.gen();
        }
        let c = hasher(&my_array);
        let (c, _) = permissible_commitment(&c, &pallas::Point::generator());
        commitments.push(c);
    }

    let mut commitments_x: Vec<Value<Fp>> = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let pt = commitments[i].to_affine().coordinates().expect("Couldn't get coordinates of a point"); 
        commitments_x.push(Value::known(*pt.x()));
    }

    let mut commitments_y: Vec<Value<Fp>> = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let pt = commitments[i].to_affine().coordinates().expect("Couldn't get coordinates of a point"); 
        commitments_y.push(Value::known(*pt.y()));
    }

    let witness = commitments_x[index].clone();
    let witness_y = commitments_y[index].clone();
    let gen = ValueCommitR::generator(&ValueCommitR);
    let generator = Ep::from(gen);
    let rerand_scalar = pallas::Scalar::from_raw([1, 2, 3, 4]);
    let rerand_pt = commitments[index].clone() + generator * rerand_scalar;
    let rerand_scalar = Value::known(rerand_scalar);
    let w_sqrt: Value<Option<Fp>> = witness_y.map(|v| v.sqrt().into());
    let w_sqrt = w_sqrt.map(|opt_fp| opt_fp.unwrap_or_default());


    let circuit = MyCircuit {
        commits_x: commitments_x,
        commits_y: commitments_y,
        witness,
        w_sqrt,
        rerand_scalar,
        k: iterations,
        index,

    };

    let mut commitments_x: Vec<Value<Fp>> = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        commitments_x.push(Value::unknown());
    }

    let mut commitments_y: Vec<Value<Fp>> = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        commitments_y.push(Value::unknown());
    }

    let witness = Value::unknown();
    let scalar: Value<Fq> = Value::unknown();

    let empty_circuit = MyCircuit {
        commits_x: commitments_x,
        commits_y: commitments_y,
        witness: witness,
        w_sqrt: witness,
        rerand_scalar: scalar,
        k: iterations,
        index,
    };

    let tmp = (rerand_pt).to_affine().coordinates().expect("Couldn't get coordinates of a point"); 
    let public_input = &[*tmp.x(), *tmp.y()];
    

    let start_time = Instant::now();
    let (params, pk) = keygen(k, empty_circuit.clone());
    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(start_time);
    println!("Elapsed keygen time: {:?}ms", elapsed_time.as_millis());

    let start_time = Instant::now();
    let proof = prover(&params, &pk, circuit, &[public_input]);
    let end_time = Instant::now();
    let elapsed_time = end_time.duration_since(start_time);
    println!("Elapsed prover time: {:?}ms", elapsed_time.as_millis()*depth);
    let proof_size = proof.len() as f64 / 1024.;
    println!("Proof size is {}kb", (depth as f64)*proof_size);

    // let start_time = Instant::now();
    // verifier(&params, pk.get_vk(), &proof, &[public_input]);
    // let end_time = Instant::now();
    // let elapsed_time = end_time.duration_since(start_time);
    // println!("Elapsed verifier time: {:?}ms", elapsed_time.as_millis());

    // let mut batch: BatchVerifier<EqAffine> = BatchVerifier::new();
    // for _ in 0..depth {
    //     batch.add_proof(vec![vec![vec![*tmp.x(), *tmp.y()]]], proof.clone());
    // }

    // let start_time = Instant::now();
    // assert!(batch.finalize(&params, pk.get_vk()));
    // let end_time = Instant::now();
    // let elapsed_time = end_time.duration_since(start_time);
    // println!(
    //     "Elapsed batch verifier time: {:?}ms",
    //     elapsed_time.as_millis()
    // );
}
