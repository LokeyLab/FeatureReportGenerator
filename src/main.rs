#![allow(non_snake_case)]
use clap::*;
use rayon::ThreadPoolBuilder;
use utils::*;

#[derive(Parser, Debug)]
#[command(name = "Feature Report Generator")]
#[command(version = "1.0")]
#[command(about = "Generates a similarity report between a experimental dataset and a reference set", long_about = None)]
struct Args {
    #[arg(short, long, help = "Path to experimental dataset (.csv format)")]
    exp_input: String,

    #[arg(short, long, help = "Path to reference dataset (.csv format)")]
    ref_input: String,

    #[arg(short, long, help = "Output file path (must end in .xlsx)")]
    outpath: String,

    #[arg(
        short,
        long,
        help = "Number of threads to use (optional: all avaliable cores avaliable if not specified)"
    )]
    threads: Option<usize>,

    #[arg(short, long, help = "index column")]
    index: usize,

    #[arg(
        short,
        long,
        help = "If distance correlation (default) or pearson r should be used"
    )]
    distance: bool,
}

fn main() {
    let args = Args::parse();

    let exp_df = read_csv(&args.exp_input).unwrap();
    let ref_df = read_csv(&args.ref_input).unwrap();

    if exp_df.width() != ref_df.width() {
        panic!("Experimental and reference sets do not have the same number of features")
    }

    if let Some(t) = args.threads {
        ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .unwrap();
    }

    let dist = !args.distance;
    let idx = args.index;
    let out_path = args.outpath;
    let rep_df = pairwise_corr_process(&exp_df, &ref_df, dist, idx);

    if let Ok(ref df) = rep_df {
        write_dataframe(df, idx, &out_path).unwrap();
    }
}
