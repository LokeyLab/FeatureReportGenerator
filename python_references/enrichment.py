import pandas as pd
from scipy.stats import fisher_exact, hypergeom
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, glob,sys 

def process_sheet(args):
    # Unpack arguments
    xls_path, sheet_name, class_dict, top_n, indexID = args
    
    # Read the specific sheet
    df = pd.read_excel(xls_path, sheet_name=sheet_name)
    
    # Consider only the top N distances
    df_sorted = df.sort_values(by=f'Exp: \"{indexID}\"').head(top_n)
    M = len(df_sorted)  # Total items considered is now the top N
    
    # Prepare the result dictionary for this sheet
    result = {'Sheet Title': indexID}
    
    # Calculate p-values for each class using both tests
    for class_name, members in class_dict.items():
        n = df_sorted[df_sorted['Reference'].isin(members)].shape[0]
        K = min(len(members), M)  # Ensure K does not exceed the number of items considered
        k = df_sorted[df_sorted['Reference'].isin(members)].head(K).shape[0]
        
        # Hypergeometric Test
        hyper_p_value = hypergeom.sf(k-1, M, n, K)
        result[f'Hypergeometric_P-Value_{class_name}'] = hyper_p_value

        # Fisher Exact Test
        contingency_table = [
            [k, K - k],
            [n - k, M - n - (K - k)]
        ]
        _, fisher_p_value = fisher_exact(contingency_table)
        result[f'Fisher_P-Value_{class_name}'] = fisher_p_value

    return result

def enrichment_analysis_parallel(workbook_path, class_dict, num_workers=4, top_n=100):
    xls = pd.ExcelFile(workbook_path)
    coversheet = pd.read_excel(xls, sheet_name='SUMMARY',index_col=0)
    # Exp: "DMSO._.NA._.A01"
    # Prepare arguments for parallel processing
    tasks = [(workbook_path, row['Tab Num'], class_dict, top_n, index) for index, row in coversheet.iterrows()]
    
    # Use ThreadPoolExecutor to process sheets in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_sheet = {executor.submit(process_sheet, task): task for task in tasks}
        for future in as_completed(future_to_sheet):
            results.append(future.result())
    
    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with the coversheet based on SheetName
    updated_coversheet = pd.merge(coversheet, results_df, on='Sheet Title')
    
    updated_coversheet.to_csv(os.path.join(os.path.dirname(workbook_path),os.path.basename(workbook_path).replace('.xlsx',"_wellEnrichments.csv")))
    
    # Save the updated DataFrame back to the workbook
    # with pd.ExcelWriter(workbook_path.replace('.xlsx','_enrichment.xlsx'), engine='xlsxwriter', mode='a', if_sheet_exists='replace') as writer:
    #     updated_coversheet.to_excel(writer, sheet_name='SUMMARY', index=False)
    return(updated_coversheet)


annots = os.path.abspath("/hb/home/alohith/KStest/reducedKey_cytoscapeAnnot.xlsx")
annotsDF = pd.read_excel(annots, sheet_name="reducedKey",index_col='IDname')

testClasses = "- STAT- VKOR- PAR- Glucokinase- Ras- HMG-CoA_Reductase- DUB- TNF- COX- AURK- Glucocorticoid_Receptor".split('- ')

class_dict = dict().fromkeys(testClasses)

class_dict.pop('')

for testClass in testClasses:
    if testClass !="":
        class_dict[testClass] = annotsDF[annotsDF['AL_CONSOLIDATED'] == testClass].index.tolist()


UMAP_noPMA_longtrain_EnrichmentAnalysis = pd.DataFrame()
for f in glob.glob("/hb/home/alohith/KStest/TM_FeatReports_MOAST/UMAP_noPMA_longtrain*.xlsx"):
    print(f)
    pname = f.split('_')[-1].replace('.xlsx','')
    df = enrichment_analysis_parallel(f, class_dict, num_workers = 28,top_n=500)
    df['Sheet Title'] = df['Sheet Title']+f"._.{pname}"
    UMAP_noPMA_longtrain_EnrichmentAnalysis = pd.concat([UMAP_noPMA_longtrain_EnrichmentAnalysis,df])

UMAP_noPMA_longtrain_EnrichmentAnalysis.to_excel("/hb/home/alohith/KStest/UMAP_noPMA_longtrain_wellEnrichments.xlsx")

del UMAP_noPMA_longtrain_EnrichmentAnalysis

UMAP_PMA_noPMA_horiztrain_EnrichmentAnalysis = pd.DataFrame()
for f in glob.glob("/hb/home/alohith/KStest/TM_FeatReports_MOAST/UMAP_PMA_noPMA_horiztrain*.xlsx"):
    print(f)
    pname = f.split('_')[-1].replace('.xlsx','')
    df = enrichment_analysis_parallel(f, class_dict, num_workers = 28,top_n=500)
    df['Sheet Title'] = df['Sheet Title']+f"._.{pname}"
    UMAP_PMA_noPMA_horiztrain_EnrichmentAnalysis = pd.concat([UMAP_PMA_noPMA_horiztrain_EnrichmentAnalysis,df])
    
UMAP_PMA_noPMA_horiztrain_EnrichmentAnalysis.to_excel('/hb/home/alohith/KStest/UMAP_PMA+noPMA_horiztrain_EnrichmentAnalysis.xlsx')


