use polars::prelude::*;
use rayon::prelude::*;

fn centered_correlation(u: &Series, v: &Series) -> f64 {
    let umu = u.mean().unwrap();
    let vmu = v.mean().unwrap();
    let u_centered = u - umu;
    let v_centered = v - vmu;
    let uv = (&u_centered * &v_centered).mean().unwrap();
    let uu = (&u_centered * &u_centered).mean().unwrap();
    let vv = (&v_centered * &v_centered).mean().unwrap();
    return 1.0 - uv / (uu * vv).sqrt();
}
