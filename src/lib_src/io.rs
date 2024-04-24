use polars::prelude::*;
use xlsxwriter::*;

/// Sanitizes a string to be a valid Excel worksheet name.
fn sanitize_worksheet_name(name: &str) -> String {
    let invalid_chars = &[':', '\\', '/', '?', '*', '[', ']'];
    let mut clean_name: String = name
        .chars()
        .filter(|c| !invalid_chars.contains(c))
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
pub fn write_dataframe(df: &DataFrame, idx: usize, outpath: &str) -> Result<(), XlsxError> {
    let workbook = Workbook::new(outpath)?;
    let nrows = df.height();
    let col_names = df.get_column_names();

    for row_idx in 0..nrows {
        let row = df.get(row_idx).unwrap(); // getting a row is actually expensive

        let sheet_name: &str = &row[idx].clone().to_string();
        let trunc_name: &str = &truncate_string(sheet_name, 30);
        let clean_name: &str = &sanitize_worksheet_name(trunc_name);

        // if let Some(w) = workbook.get_worksheet(trunc_name).unwrap() {
        //     continue; // Skip adding a new worksheet if it already exists
        // }

        println!("{}", trunc_name);
        let mut worksheet = workbook.add_worksheet(Some(clean_name))?;
        let _ = worksheet.write_string(0, 0, "Reference", None);
        let _ = worksheet.write_string(0, 1, &format!("Exp: {}", sheet_name), None);

        for (col_idx, val) in row.iter().enumerate().skip(idx + 1) {
            // let v = to_f64(val).unwrap();
            let v = match val {
                // type conversion to float
                AnyValue::Float64(f) => *f,
                _ => panic!("not a float value"),
            };

            let _ = worksheet.write_string(col_idx as u32, 0, col_names[col_idx], None);
            let _ = worksheet.write_number(col_idx as u32, 1, v, None);
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

    #[test]
    fn test_writer() -> Result<(), PolarsError> {
        let df = create_random_dataframe(10, 12, true)?;
        println!("{df}");

        write_dataframe(&df, 0, "yrjhdyf.xlsx").unwrap();
        return Ok(());
    }

    #[test]
    fn test_full_set() -> Result<(), PolarsError> {
        let exp_df = read_csv("/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uMData/TargetMol_10uM2PMA_1_HD.csv").unwrap();
        // let ref_df = read_csv("/mnt/c/Users/derfelt/Desktop/LokeyLabFiles/TargetMol/Datasets/10uM/10uM_concats_complete/TargetMol_10uM_PMA_plateConcat_DEADS_DROPPED_HD.csv").unwrap();
        // if exp_df.width() != ref_df.width() {
        //     panic!("Experimental and reference sets do not have the same number of features")
        // }

        // let res_df = pairwise_corr_process(&exp_df, &ref_df, true, 0).unwrap();
        write_dataframe(&exp_df, 0, "test.xlsx").unwrap();
        Ok(())
    }
}
