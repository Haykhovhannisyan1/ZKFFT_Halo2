use ff::BatchInverter;
use group::{Curve, Group};
use halo2_proofs::transcript::{Blake2bWrite, Challenge255, ChallengeScalar, EncodedChallenge, TranscriptWrite};
use halo2_proofs::arithmetic::{CurveAffine, Field, best_multiexp};
use pasta_curves::arithmetic::CurveExt;
use pasta_curves::{EpAffine,pallas, Ep, Fq};
use rand::{rngs::OsRng, Rng};
use std::time::Instant;

pub struct WipWitness {
    a: Vec<Fq>,
    b: Vec<Vec<Fq>>,
    alpha: Fq,
}

pub struct WipProof {
    L: Vec<Ep>,
    R: Vec<Ep>,
    A: Ep,
    B: Ep,
    r_answer: Fq,
    s_answer: Vec<Fq>,
    delta_answer: Fq,
}

#[derive(PartialEq, Debug)]
pub enum P{
    Point(Ep),
    Terms(Vec<(Fq, Ep)>),
}

pub fn inner_product(a: &[Fq], b: &[Fq]) -> Fq {
    assert_eq!(a.len(), b.len());

    let mut acc = Fq::from(0);
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

pub fn multiexp(p: &P) -> Ep {
    match p {
        P::Point(p) => {
            *p
        }
        P::Terms(v) => {
            let (coeffs, bases): (Vec<Fq>, Vec<Ep>) = v.into_iter().cloned().unzip();
            let mut new_bases = Vec::with_capacity(bases.len());
            for b in bases.iter() {
                let tmp = b.to_affine().coordinates().expect("Couldn't get coordinates of a point");
                new_bases.push(EpAffine::from_xy(*tmp.x(), *tmp.y()).expect("Couldn't construct point from coordinates")); 
            }
            best_multiexp::<pasta_curves::EpAffine>(&coeffs, &new_bases)
        }
    }
}


fn split_vector_in_half<T: Clone>(vec: Vec<T>) -> (Vec<T>, Vec<T>) {
    let mid = vec.len() / 2 + vec.len() % 2; // calculate midpoint, extra element goes into the first half if odd length
    let (first_half, second_half) = vec.split_at(mid);
    (first_half.to_vec(), second_half.to_vec()) // convert slices to vectors
}

fn challenge_products(challenges: &[(Fq, Fq)]) -> Vec<Fq> {
    let mut products = vec![Fq::ONE; 1 << challenges.len()];

    if !challenges.is_empty() {
      products[0] = challenges[0].1;
      products[1] = challenges[0].0;

      for (j, challenge) in challenges.iter().enumerate().skip(1) {
        let mut slots = (1 << (j + 1)) - 1;
        while slots > 0 {
          products[slots] = products[slots / 2] * challenge.0;
          products[slots - 1] = products[slots / 2] * challenge.1;

          slots = slots.saturating_sub(2);
        }
      }

      // Sanity check since if the above failed to populate, it'd be critical
      for product in &products {
        debug_assert!(!bool::from(product.is_zero()));
      }
    }

    products
}

fn transcript_A_B<
    E: EncodedChallenge<pallas::Affine>,
    T: TranscriptWrite<pallas::Affine, E>
>(transcript: &mut T, A: Ep, B: Ep) -> Fq {
    transcript.write_point(A.into());
    transcript.write_point(B.into());

    let e: ChallengeScalar<pallas::Affine, T> = transcript.squeeze_challenge_scalar();
    if bool::from(e.is_zero()) {
      panic!("zero challenge in final WIP round");
    }
    *e
}

fn transcript_L_R<
    E: EncodedChallenge<pallas::Affine>,
    T: TranscriptWrite<pallas::Affine, E>
>(transcript: &mut T, L: Ep, R: Ep) -> Fq {
    transcript.write_point(L.into());
    transcript.write_point(R.into());

    let e: ChallengeScalar<pallas::Affine, T> = transcript.squeeze_challenge_scalar();
    if bool::from(e.is_zero()) {
      panic!("zero challenge in final WIP round");
    }
    *e
}

fn next_G_H<
    E: EncodedChallenge<pallas::Affine>,
    T: TranscriptWrite<pallas::Affine, E>
>(
    transcript: &mut T,
    g_bold1: Vec<Ep>,
    g_bold2: Vec<Ep>,
    h_bold1: Vec<Vec<Ep>>,
    h_bold2: Vec<Vec<Ep>>,
    L: Ep,
    R: Ep,
    ) -> (Fq, Fq, Fq, Fq, Vec<Ep>, Vec<Vec<Ep>>) {
    assert_eq!(g_bold1.len(), g_bold2.len());
    for (h1, h2) in h_bold1.iter().zip(h_bold2.iter()) {
        assert_eq!(h1.len(), h2.len());
        assert_eq!(g_bold1.len(), h1.len());
    }

    let e = transcript_L_R(transcript, L, R); 
    let inv_e = e.invert().unwrap();

    let mut new_g_bold = Vec::with_capacity(g_bold1.len());
    for g_bold in g_bold1.iter().cloned().zip(g_bold2.iter().cloned()) {
        let tmp: P = P::Terms(vec![(inv_e, g_bold.0), (e, g_bold.1)]);
        new_g_bold.push(multiexp(&tmp));
    }

    let mut new_h_bold = Vec::with_capacity(h_bold1.len());
    for (h_bold1_vec, h_bold2_vec) in h_bold1.iter().zip(h_bold2.iter()) {
        let mut inner_results = Vec::with_capacity(h_bold1_vec.len());
        for (h1, h2) in h_bold1_vec.iter().cloned().zip(h_bold2_vec.iter().cloned()) {
            let tmp: P = P::Terms(vec![(e, h1), (inv_e, h2)]);
            inner_results.push(multiexp(&tmp));
        }
        new_h_bold.push(inner_results);
    }

    let e_square = e.square();
    let inv_e_square = inv_e.square();

    (e, inv_e, e_square, inv_e_square, new_g_bold, new_h_bold)
}

pub fn prove<
    E: EncodedChallenge<pallas::Affine>,
    T: TranscriptWrite<pallas::Affine, E>
>(
    transcript: &mut T,
    witness: WipWitness,
    generators_g: Vec<Ep>,
    generators_h: Vec<Vec<Ep>>,
    generator_g: Vec<Ep>,
    generator_h: Ep,
    p: P
) -> WipProof {
    let mut rng = OsRng;

    // Check P has the expected relationship
    if let P::Point(p) = &p {
        let mut p_terms = witness.a
        .iter()
        .copied()
        .zip(generators_g.iter().copied())
        .collect::<Vec<_>>();
    
        for (witness_b_vec, generators_h_vec) in witness.b.iter().zip(generators_h.iter()) {
            let additional_terms: Vec<(Fq, Ep)> = witness_b_vec
                .iter()
                .copied()
                .zip(generators_h_vec.iter().copied())
                .collect::<Vec<_>>();
            p_terms.extend(additional_terms);
        }
        let mut additional_terms: Vec<(Fq, Ep)> = vec![];
        for (b_i, g_i) in witness.b.iter().zip(generator_g.iter()) {
            additional_terms.push((inner_product(&witness.a, b_i),*g_i));
        }
        p_terms.extend(additional_terms);
        p_terms.push((witness.alpha, generator_h));
        assert_eq!(multiexp(&P::Terms(p_terms)), *p);
    }

    let mut g_bold = generators_g.clone();
    let mut h_bold = generators_h.clone();

    let mut a = witness.a.clone();
    let mut b = witness.b.clone();
    let mut alpha = witness.alpha;
    for b_i in b.iter() {
        assert_eq!(a.len(), b_i.len());
    }

    // // From here on, g_bold.len() is used as n
    assert_eq!(g_bold.len(), a.len());

    let mut L_vec : Vec<Ep> = vec![];
    let mut R_vec : Vec<Ep> = vec![];

    // // else n > 1 case from figure 1
    while g_bold.len() > 1 {
        let b_clone = b.clone();
        let (a1, a2) = split_vector_in_half(a.clone());
        let (b1, b2): (Vec<Vec<_>>, Vec<Vec<_>>) = b_clone
            .into_iter()
            .map(split_vector_in_half)
            .unzip();
        let (g_bold1, g_bold2) = split_vector_in_half(g_bold.clone());
        let (h_bold1, h_bold2): (Vec<Vec<_>>, Vec<Vec<_>>) = h_bold
        .into_iter()
        .map(split_vector_in_half)
        .unzip();

        let n_hat = g_bold1.len();
        assert_eq!(a1.len(), n_hat);
        assert_eq!(a2.len(), n_hat);
        for b_i in b1.iter() {
            assert_eq!(n_hat, b_i.len());
        }
        for b_i in b2.iter() {
            assert_eq!(n_hat, b_i.len());
        }
        assert_eq!(g_bold1.len(), n_hat);
        assert_eq!(g_bold2.len(), n_hat);
        for h_i in h_bold1.iter() {
            assert_eq!(n_hat, h_i.len());
        }
        for h_i in h_bold2.iter() {
            assert_eq!(n_hat, h_i.len());
        }

        let d_l = Fq::random(rng);
        let d_r = Fq::random(rng); 
    

        let mut c_l : Vec<Fq> = vec![];
        let mut c_r : Vec<Fq> = vec![];

        for b_i in b2.iter() {
            let tmp = inner_product(&a1, b_i);
            c_l.push(tmp);
        }
        for b_i in b1.iter() {
            let tmp = inner_product(&a2, b_i);
            c_r.push(tmp);
        }

        let mut L_terms: Vec<(Fq, Ep)> = a1.iter().copied()
            .zip(g_bold2.iter().copied())
            .chain(c_l.iter().copied().zip(generator_g.iter().copied()))
            .collect::<Vec<_>>();
        L_terms.push((d_l, generator_h));
        for (witness_b_vec, generators_h_vec) in b2.iter().zip(h_bold1.iter()) {
            let additional_terms = witness_b_vec
                .iter()
                .copied()
                .zip(generators_h_vec.iter().copied())
                .collect::<Vec<_>>();
            L_terms.extend(additional_terms);
        }
        let L = multiexp(&P::Terms(L_terms));
        L_vec.push(L);


        let mut R_terms: Vec<(Fq, Ep)> = a2.iter().copied()
            .zip(g_bold1.iter().copied())
            .chain(c_r.iter().copied().zip(generator_g.iter().copied()))
            .collect::<Vec<_>>();
        R_terms.push((d_r, generator_h));
        for (witness_b_vec, generators_h_vec) in b1.iter().zip(h_bold2.iter()) {
            let additional_terms = witness_b_vec
                .iter()
                .copied()
                .zip(generators_h_vec.iter().copied())
                .collect::<Vec<_>>();
            R_terms.extend(additional_terms);
        }
        let R = multiexp(&P::Terms(R_terms));
        R_vec.push(R);

        let (e, inv_e, e_square, inv_e_square);
        (e, inv_e, e_square, inv_e_square, g_bold, h_bold) =
            next_G_H(transcript, g_bold1, g_bold2, h_bold1, h_bold2, L, R);


        let tmp1 : Vec<Fq> = a1.into_iter().map(|x| x * e).collect();
        let tmp2 : Vec<Fq> = a2.into_iter().map(|x| x * inv_e).collect();
        a = tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect();

        b = vec![];
        for (b1_i, b2_i ) in b1.iter().zip(b2.iter()) {
            let  tmp1 : Vec<Fq> = b1_i.into_iter().map(|x| x * inv_e).collect();
            let tmp2 : Vec<Fq> = b2_i.into_iter().map(|x| x * e).collect();
            b.push(tmp1.iter().zip(tmp2.iter()).map(|(&a, &b)| a + b).collect());
        }


        alpha += (d_l * e_square) + (d_r * inv_e_square);

        debug_assert_eq!(g_bold.len(), a.len());
        for h_i in h_bold.iter() {
            debug_assert_eq!(g_bold.len(), h_i.len());
        }
        for b_i in b.iter() {
            debug_assert_eq!(g_bold.len(), b_i.len());
        }
    }

    // // n == 1 case from figure 1
    assert_eq!(g_bold.len(), 1);
    for h_i in h_bold.iter() {
        assert_eq!(h_i.len(), 1);
    }
    assert_eq!(a.len(), 1);
    for b_i in b.iter() {
        assert_eq!(b_i.len(), 1);
    }
    
        //////////////////////////////////////////////////////////

    let r = Fq::random(rng);
    let mut s: Vec<Fq> = vec![];
    for _i in h_bold.iter() {
        s.push(Fq::random(rng));
    }
    let delta = Fq::random(rng); 
    let long_n = Fq::random(rng); 

    let mut h_terms : Vec<(Fq, Ep)> = vec![];
    for (s_i, h_i) in s.iter().zip(h_bold.iter()) {
        h_terms.push((*s_i, h_i[0]))
    }
    let mut g_terms : Vec<(Fq, Ep)> = vec![];
    for ((s_i, g_i), b_i) in s.iter().zip(generator_g.iter()).zip(b.iter()) {
        g_terms.push(((r * b_i[0]) + (*s_i * a[0]), *g_i))
    }

    let mut A_terms: Vec<(Fq, Ep)> =
      vec![(r, g_bold[0]), (delta, generator_h)];
    A_terms.extend(h_terms);
    A_terms.extend(g_terms);
    let A = multiexp(&P::Terms(A_terms));
    // A_terms.zeroize();

    let mut g_terms : Vec<(Fq, Ep)> = vec![];
    for (s_i, g_i) in s.iter().zip(generator_g.iter()) {
        g_terms.push((r * *s_i, *g_i));
    }

    let mut B_terms: Vec<(Fq, Ep)> = vec![(long_n, generator_h)];
    B_terms.extend(g_terms);
    let B = multiexp(&P::Terms(B_terms));
    // B_terms.zeroize();

    let e = transcript_A_B(transcript, A, B);

    let r_answer = r + (a[0] * e);
    let mut s_answer: Vec<Fq> = vec![];
    for (s_i, b_i) in s.iter().zip(b.iter()) {
        s_answer.push(s_i + (b_i[0] * e));
    }
    let delta_answer = long_n + (delta * e) + (alpha * e.square());

    WipProof { L: L_vec, R: R_vec, A, B, r_answer, s_answer, delta_answer }
  }


pub fn verify<
    E: EncodedChallenge<pallas::Affine>,
    T: TranscriptWrite<pallas::Affine, E>
    >(
    transcript: &mut T,
    proof: WipProof,
    generators_g: Vec<Ep>,
    generators_h: Vec<Vec<Ep>>,
    generator_g: Vec<Ep>,
    generator_h: Ep,
    p: P,
) {
    // Verify the L/R lengths
    {
        let mut lr_len = 0;
        while (1 << lr_len) < generators_g.len() {
        lr_len += 1;
        }
        assert_eq!(proof.L.len(), lr_len);
        assert_eq!(proof.R.len(), lr_len);
        assert_eq!(generators_g.len(), 1 << lr_len);
    }

    let mut P_terms = match p {
        P::Point(point) => vec![(Fq::ONE, point)],
        P::Terms(terms) => terms,
    };
    // P_terms.reserve(6 + (2 * generators_g.len()) + proof.L.len());

    let mut challenges = Vec::with_capacity(proof.L.len());
    let product_cache = {
        let mut es = Vec::with_capacity(proof.L.len());
        for (L, R) in proof.L.iter().zip(proof.R.iter()) {
            es.push(transcript_L_R(transcript, *L, *R));
        }

        let mut inv_es = es.clone();
        let mut scratch = vec![Fq::ZERO; es.len()];
        BatchInverter::invert_with_external_scratch(
            &mut inv_es,
            &mut scratch
        );
        drop(scratch);

        assert_eq!(es.len(), inv_es.len());
        assert_eq!(es.len(), proof.L.len());
        assert_eq!(es.len(), proof.R.len());
        for ((e, inv_e), (L, R)) in
            es.drain(..).zip(inv_es.drain(..)).zip(proof.L.iter().zip(proof.R.iter()))
        {
        debug_assert_eq!(e.invert().unwrap(), inv_e);

        challenges.push((e, inv_e));

        let e_square = e.square();
        let inv_e_square = inv_e.square();
        P_terms.push((e_square, *L));
        P_terms.push((inv_e_square, *R));
        }

        challenge_products(&challenges)
    };

    let e = transcript_A_B(transcript, proof.A, proof.B);
    let neg_e_square = -e.square();

    let mut multiexp_var = P_terms;
    // multiexp_var.reserve(4 + (2 * generators_g.len()));
    for (scalar, _) in multiexp_var.iter_mut() {
        *scalar *= neg_e_square;
    }

    let re = proof.r_answer * e;
    for i in 0 .. generators_g.len() {
        let mut scalar = product_cache[i] * re;
        multiexp_var.push((scalar, generators_g[i].clone()));
    }

    for i in 0 .. generators_h.len() {
        let se = proof.s_answer[i] * e;
        for j in 0 .. generators_h[i].len() {
            multiexp_var.push((
                se * product_cache[product_cache.len() - 1 - j],
                generators_h[i][j].clone(),
            ));
        }
    }


    multiexp_var.push((-e, proof.A));
    let mut i = 0;
    for g_i in generator_g.iter() {
        multiexp_var.push((proof.r_answer * proof.s_answer[i], *g_i));
        i += 1;
    }
    multiexp_var.push((proof.delta_answer, generator_h));
    multiexp_var.push((-Fq::ONE, proof.B));

    assert_eq!(multiexp(&P::Terms(multiexp_var)), Ep::identity());
}

fn gens() -> (Vec<pallas::Point>, Vec<Vec<pallas::Point>>, Vec<pallas::Point>, pallas::Point) {
    let mut gens_g = Vec::with_capacity(4);
    let mut gens_h = Vec::with_capacity(4);
    let mut gen_g = Vec::with_capacity(4);
    let hasher = pallas::Point::hash_to_curve("GENERATORS");

    for _i in 0..4 {
        let mut my_array: [u8; 11] = [0; 11];

        let mut rng = rand::thread_rng();
        for i in 0..11 {
            my_array[i] = rng.gen();
        }
        let c = hasher(&my_array);
        gens_g.push(c);
    }

    for _i in 0..4 {
        let mut my_array: [u8; 11] = [0; 11];

        let mut rng = rand::thread_rng();
        for i in 0..11 {
            my_array[i] = rng.gen();
        }
        let c = hasher(&my_array);
        gen_g.push(c);
    }

    for _j in 0..4 {
        let mut temp_gens = Vec::with_capacity(4);
        for _i in 0..4 {
            let mut my_array: [u8; 11] = [0; 11];
    
            let mut rng = rand::thread_rng();
            for i in 0..11 {
                my_array[i] = rng.gen();
            }
            let c = hasher(&my_array);
            temp_gens.push(c);
        }
        gens_h.push(temp_gens);
    }


    let mut my_array: [u8; 11] = [0; 11];

    let mut rng = rand::thread_rng();
    for i in 0..11 {
        my_array[i] = rng.gen();
    }
    let c = hasher(&my_array);
    let gen_h = c;

    return (gens_g, gens_h, gen_g, gen_h);
}

fn main() {
    let mut transcript = Blake2bWrite::<_, pallas::Affine, Challenge255<_>>::init(vec![]);
    let w = WipWitness{
        a: vec![pallas::Scalar::from(2), pallas::Scalar::from(3), pallas::Scalar::from(2), pallas::Scalar::from(2)],
        b: vec![
            vec![pallas::Scalar::from(1), pallas::Scalar::from(0), pallas::Scalar::from(0), pallas::Scalar::from(0)],
            vec![pallas::Scalar::from(0), pallas::Scalar::from(1), pallas::Scalar::from(0), pallas::Scalar::from(0)],
            vec![pallas::Scalar::from(0), pallas::Scalar::from(0), pallas::Scalar::from(1), pallas::Scalar::from(0)],
            vec![pallas::Scalar::from(0), pallas::Scalar::from(0), pallas::Scalar::from(0), pallas::Scalar::from(1)],
        ],
        alpha: pallas::Scalar::from(5)
    };

    let (gens_g, gens_h, gen_g, gen_h) = gens();
    let ip = vec![inner_product(&w.a, &w.b[0]), inner_product(&w.a, &w.b[1]), inner_product(&w.a, &w.b[2]), inner_product(&w.a, &w.b[3])];
    // let ip = vec![inner_product(&w.a, &w.b[0]), inner_product(&w.a, &w.b[1]), inner_product(&w.a, &w.b[2]), Fq::from(4)];
    let mut commit = Ep::identity();
    for i in 0..4 {
        commit += gens_g[i] * w.a[i];
    }
    for i in 0..4 {
        for j in 0..4 {
            commit += gens_h[i][j] * w.b[i][j];
        }
    }

    for i in 0..4 {
        commit += gen_g[i]*ip[i];
    }
    commit += gen_h*w.alpha; 
    let now: Instant = Instant::now();
    let proof = prove(&mut transcript, w, gens_g.clone(), gens_h.clone(), gen_g.clone(), gen_h.clone(), P::Point(commit));
    println!("Generated the proof in {}ms", now.elapsed().as_millis());
    let mut transcript = Blake2bWrite::<_, pallas::Affine, Challenge255<_>>::init(vec![]);
    let now: Instant = Instant::now();
    verify(&mut transcript, proof, gens_g, gens_h, gen_g, gen_h, P::Point(commit));
    println!("Verified the proof in {}ms", now.elapsed().as_millis());

}

// fn main() {
//     let mut transcript = Blake2bWrite::<_, pallas::Affine, Challenge255<_>>::init(vec![]);
//     let w = WipWitness{
//         a: vec![pallas::Scalar::from(2), pallas::Scalar::from(3), pallas::Scalar::from(2), pallas::Scalar::from(2)],
//         b: vec![
//             vec![pallas::Scalar::from(1), pallas::Scalar::from(0), pallas::Scalar::from(0), pallas::Scalar::from(0)],
//         ],
//         alpha: pallas::Scalar::from(5)
//     };

//     let (gens_g, gens_h, gen_g, gen_h) = gens();
//     let gens_h = vec![gens_h[0].clone()];
//     let gen_g = vec![gen_g[0]];
//     let ip = vec![inner_product(&w.a, &w.b[0])];
//     let mut commit = Ep::identity();
//     for i in 0..4 {
//         commit += gens_g[i] * w.a[i];
//     }
//     for i in 0..1 {
//         for j in 0..1 {
//             commit += gens_h[i][j] * w.b[i][j];
//         }
//     }

//     for i in 0..1 {
//         commit += gen_g[i]*ip[i];
//     }
//     commit += gen_h*w.alpha; 

//     let proof = prove(&mut transcript, w, gens_g.clone(), gens_h.clone(), gen_g.clone(), gen_h.clone(), P::Point(commit));
//     let mut transcript = Blake2bWrite::<_, pallas::Affine, Challenge255<_>>::init(vec![]);
//     verify(&mut transcript, proof, gens_g, gens_h, gen_g, gen_h, P::Point(commit))
// }
