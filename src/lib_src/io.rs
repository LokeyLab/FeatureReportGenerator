#![allow(dead_code)]
use polars::prelude::*;
use std::fmt;
use xlsxwriter::*;

#[derive(Debug)]
pub enum IoError {
    XlsxError(XlsxError),
    PolarsError(polars::error::PolarsError),
    InvalidData(String),
}

// Implement std::fmt::Display for DataFrameError
impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IoError::XlsxError(ref err) => write!(f, "Xlsx Error: {}", err),
            IoError::PolarsError(ref err) => write!(f, "Polars Error: {}", err),
            IoError::InvalidData(ref desc) => write!(f, "Invalid data: {}", desc),
        }
    }
}

impl From<polars::error::PolarsError> for IoError {
    fn from(err: polars::error::PolarsError) -> Self {
        IoError::PolarsError(err)
    }
}

impl From<XlsxError> for IoError {
    fn from(err: XlsxError) -> Self {
        IoError::XlsxError(err)
    }
}

/// Sanitizes a string to be a valid Excel worksheet name.
fn sanitize_worksheet_name(name: &str) -> String {
    let invalid_chars = &[':', '\\', '/', '?', '*', '[', ']'];
    let mut clean_name: String = name
        .chars()
        .map(|c| if invalid_chars.contains(&c) { '_' } else { c })
        .collect();

    // Truncate to maximum length for Excel sheet names (31 characters)
    if clean_name.len() > 31 {
        clean_name.truncate(31);
    }

    // Replace empty names with a default name
    if clean_name.is_empty() {
        clean_name = "DefaultName".to_string();
    }

    clean_name
}

/// fits characters into max_len
fn truncate_string(s: &str, max_len: usize) -> String {
    match s.char_indices().nth(max_len) {
        Some((idx, _)) => s[..idx].to_string(),
        None => s.to_string(),
    }
}

/// Reads a csv and returns a polars dataframe
pub fn read_csv(file_path: &str) -> Result<DataFrame, PolarsError> {
    CsvReader::from_path(file_path)?.has_header(true).finish()
}

/// Writes a dataframe into standard feature report format
pub fn write_dataframe(
    corrdist_df: &DataFrame,
    pearsonr_df: &DataFrame,
    idx: usize,
    outpath: &str,
) -> Result<(), IoError> {
    let workbook = Workbook::new(outpath)?;
    let nrows = corrdist_df.height();
    let col_names = corrdist_df.get_column_names();

    if nrows != pearsonr_df.height() {
        return Err(IoError::InvalidData(
            "Dataframes are different sizes!".to_string(),
        ));
    }

    //formatting stuff
    let mut bold_format = Format::new();
    bold_format.set_bold();

    // making table of contents
    let mut tab_of_conts = workbook.add_worksheet(Some(&"SUMMARY"))?;
    let idx_col = &corrdist_df.get_columns()[idx];
    let _ = tab_of_conts.write_string(0, 0, "Sheet Title", Some(&bold_format));
    let _ = tab_of_conts.write_string(0, 1, "Tab Num", Some(&bold_format));
    for (row_idx, row) in idx_col.iter().enumerate() {
        let curr_idx = row_idx + 1;

        let _ = tab_of_conts.write_string(curr_idx as u32, 1, &format!("{}", curr_idx), None);

        let val = row.get_str().unwrap();
        let _ = tab_of_conts.write_string(curr_idx as u32, 0, val, Some(&bold_format));
    }

    for row_idx in 0..nrows {
        let corrdist_row = corrdist_df.get(row_idx).unwrap(); // getting a row is actually expensive
        let pearson_row = pearsonr_df.get(row_idx).unwrap();

        let sheet_name: &str = &corrdist_row[idx].clone().to_string();

        let mut worksheet = workbook.add_worksheet(Some(&format!("{}", row_idx + 1)))?;
        let _ = worksheet.write_string(0, 0, "Reference", Some(&bold_format));
        let _ = worksheet.write_string(0, 1, "Correlation Distance", Some(&bold_format));
        let _ = worksheet.write_string(0, 2, "Pearson R", Some(&bold_format));
        let _ = worksheet.write_string(0, 3, &format!("Exp: {}", sheet_name), Some(&bold_format));

        for (col_idx, (dist_val, pearson_val)) in corrdist_row
            .iter()
            .zip(pearson_row.iter())
            .enumerate()
            .skip(idx + 1)
        {
            let corrdist_v = match dist_val {
                AnyValue::Float64(f) => *f,
                _ => panic!("not a float value"),
            };

            let pearson_v = match pearson_val {
                AnyValue::Float64(f) => *f,
                _ => panic!("not a float"),
            };

            let _ =
                worksheet.write_string(col_idx as u32, 0, col_names[col_idx], Some(&bold_format));
            let _ = worksheet.write_number(col_idx as u32, 1, corrdist_v, None);
            let _ = worksheet.write_number(col_idx as u32, 2, pearson_v, None);
        }
    }

    let _ = workbook.close();
    return Ok(());
}

#[cfg(test)]
mod test_io {

    use super::*;
    use rand::Rng;

    fn create_random_dataframe(
        nrows: i32,
        ncols: usize,
        dummy_index: bool,
    ) -> Result<DataFrame, PolarsError> {
        let mut rng = rand::thread_rng();

        // Create a Vec of Series, each with random f64 values
        let mut series_vec = Vec::with_capacity(ncols + if dummy_index { 1 } else { 0 });

        // If dummy_index is true, create an "index" column with sequential integers
        if dummy_index {
            let index_values: Vec<String> = (0..nrows).map(|i| i.to_string()).collect(); // Generating indices
            let index_series = Series::new("index", &index_values);
            series_vec.push(index_series); // Adding the index series as the first column
        }

        // Generate random f64 columns
        for i in 0..ncols {
            let values: Vec<f64> = (0..nrows).map(|_| rng.gen()).collect();
            let series = Series::new(&format!("col{}", i), &values);
            series_vec.push(series);
        }

        // Combine all Series into a DataFrame
        let df = DataFrame::new(series_vec)?;

        // Get the list of column names, sort them, and then select columns in sorted order
        // let mut column_names = df.get_column_names();
        // column_names.sort(); // Sorts column names alphabetically
        // df = df.select(&column_names)?; // Reorder DataFrame based on sorted column names

        Ok(df)
    }

    // #[test]
    // fn test_writer() -> Result<(), PolarsError> {
    //     let df = create_random_dataframe(10, 12, true)?;
    //     println!("{df}");

    //     // write_dataframe(&df, 0, "yrjhdyf.xlsx").unwrap();
    //     return Ok(());
    // }

    // #[test]
    // fn test_full_set() -> Result<(), PolarsError> {
    //     let exp_df = read_csv("/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uMData/TargetMol_10uM2PMA_1_HD.csv").unwrap();
    //     // let ref_df = read_csv("/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_PMA_plateConcat_DEADS_DROPPED_HD.csv").unwrap();
    //     // if exp_df.width() != ref_df.width() {
    //     //     panic!("Experimental and reference sets do not have the same number of features")
    //     // }

    //     // let res_df = pairwise_corr_process(&exp_df, &ref_df, true, 0).unwrap();
    //     // write_dataframe(&exp_df, 0, "test.xlsx").unwrap();
    //     Ok(())
    // }
}
