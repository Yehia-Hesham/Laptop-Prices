def Clean_data_pt1(df,y_train = False, calc_price_stats = False,keep_ID = False,Deployment = False):
    """Cleans the original 'laptop_price' dataset and restructes its columns.
    
    === parameters ===
    - df (pandas Dataframe): dataframe of laptop_prices
    - y_train (Boolean) : dataframe 'df' includes a y_column 'Price_euros'.
    
    = Returns : structed dataframe ready for the next function."""

    import re
    import pandas as pd
    df = df.rename(columns={'Cpu':'cpu_name'})
    
    #=== Processing Price brackets (if y_train = True) ==============================================
    # calculate relevant statistics 
    if y_train:
        if calc_price_stats:
            Price_std = df.Price_euros.std() * 0.7
            Price_mean = df.Price_euros.mean()
        else:
            Price_std = 489.3
            Price_mean = 1123.6
            
        lower = Price_mean - Price_std
        higher = Price_mean + Price_std

        def laptop_price_range(row):
            """creates 'Price_bracket' column (based on 'Price_euros'), a new category useful for future analysis."""
            
            if row['Price_euros'] <= lower :
                row['Price_bracket'] = 'Budget'
            elif (row['Price_euros'] > lower) & (row['Price_euros'] <= higher):
                row['Price_bracket'] = 'Mid Range'
            else:
                row['Price_bracket'] = 'High End'
            return row
        
        df = df.apply(laptop_price_range,axis = 1)  
        
    #=== Processing Storage =========================================================================
    #- stroage types
    df['Memory_SSD'] = (((df['Memory'].str.find('SSD') +1).values) > 0) #ex: returns bool if it finds 'SSD'
    df['Memory_HDD'] = (((df['Memory'].str.find('HDD') +1).values) > 0)
    df['Memory_Flash_Storage'] = (((df['Memory'].str.find('Flash Storage') +1).values) > 0)
    df['Memory_Hybrid'] = (((df['Memory'].str.find('Hybrid') +1).values) > 0)
    
    df[['Memory_SSD','Memory_HDD','Memory_Flash_Storage',
        'Memory_Hybrid']]= df[['Memory_SSD','Memory_HDD','Memory_Flash_Storage',
                               'Memory_Hybrid']].replace({True:1,False:0})  # maps {True:1,False:0} for ML model later
    
    #- primary and secondary storage values 
    memory = []
    for sentence in df['Memory']:
        #returns a list with memory sizes (ex: 250 SSD, 1TB HDD ==> [250,1])
        items = [float(s) for s in re.findall(r'-?\d+\.?\d*', sentence)] 
        memory.append(items)
    memory = pd.DataFrame(memory,columns=['memory_Primary','memory_Secondary']) #creates dataframe_memory
    
    #adjusts memory units (ex:1TB ==> 1000 GB)
    memory['memory_Primary'] = memory['memory_Primary'].apply(lambda x: x*1000 if x<=13 else x)  
    memory['memory_Secondary'] = memory['memory_Secondary'].apply(lambda x: x*1000 if x<=13 else 0)
    
    df = pd.concat([df,memory], axis = 1) #merge new data into original dataset
    
    #=== Processig Screen ==============================================================================
    
    #- Screen_Resolution
    def get_resolution(row):
        """processes ScreenResolution into 3 columns."""
        row['Screen_Resolution'] = row['ScreenResolution'].split(" ")[-1]
        resolution_list = row['Screen_Resolution'].split("x")
        row['Screen_Resolution_W'] = resolution_list[0] 
        row['Screen_Resolution_L'] = resolution_list[1] 
        return row
    
    df = df.apply(get_resolution,axis = 1)
    
    #Screen_Panel types (IPS,Touchscreen,Retina...etc.) into columns with Boolean values
    df['Screen_Panel_IPS'] = (((df['ScreenResolution'].str.find('IPS') +1).values) > 0)
    df['Screen_Touchscreen'] = (((df['ScreenResolution'].str.find('Touchscreen') +1).values) > 0)
    df['Screen_Retina_Display'] = (((df['ScreenResolution'].str.find('Retina') +1).values) > 0)

    for col in ['Screen_Panel_IPS','Screen_Touchscreen','Screen_Retina_Display']:
        df[col] = df[col].map({True:1,False:0})

    #=== Processig Others (weight, Ram, OPSys and Type) ==================================================
    #weight
    df['Weight_kg'] = df['Weight'].str.replace("kg","")
    df['Weight_kg'] = df['Weight_kg'].astype(float)

    #Ram
    df['Ram_GB'] = df['Ram'].str.replace("GB","")
    df['Ram_GB'] = df['Ram_GB'].astype(int)
    
    #reducing under represented/irrelevant categories
    df['OpSys'] = df['OpSys'].replace({'Windows 10 S':'Windows 10','Mac OS X':'macOS'}) 
    
    df['TypeName'] = df['TypeName'].replace({'Netbook':'Notebook'}) #'Netbooks' are a type of 'Notebooks'
    
    #=== Processing Company =================================================================================
    
    #reducing under represented/irrelevant categories
    for other_1 in ['Mediacom','Microsoft','Razer','Microsoft','Google']:
        df['Company'] = df['Company'].replace({other_1:'other (european/American)'})
    for other_2 in ['Xiaomi','Chuwi','Huawei'] :
        df['Company'] = df['Company'].replace({other_2:'other (Chinease/Korean)'})
    for other_3 in ['Vero','Fujitsu']:
        df['Company'] = df['Company'].replace({other_3:'other'})     
     
    #=== Procesing CPU/GPU Brands ============================================================================
    
    df = df.rename(columns={'Gpu':'gpu_name'})
    
    def get_brands(row):
        row['cpu_brand'] = row['cpu_name'].split(" ").pop(0)
        row['gpu_brand'] = row['gpu_name'].split(" ").pop(0)
        
        return row
    
    df = df.apply(get_brands,axis = 1)
    
    #Get_Dummies
    # cols = ['Company','TypeName','OpSys']
    # df = pd.get_dummies(df, columns=cols)

    #drop extra
    df = df.drop(['Screen_Resolution','ScreenResolution','Weight','Ram','Memory','Product'],axis=1)
    # df = df.drop(['Company_other','TypeName_Workstation','OpSys_No OS'],axis = 1) #cpu_name

    # if Deployment:
    #     df = pd.get_dummies(df, columns=['cpu_brand','gpu_brand', 'price_bracket'])
        
    if not keep_ID:
        df = df.drop('laptop_ID',axis=1)

    return df
#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

def get_keys(df_laptops,df_GPUs,df_CPUs,print_pairs = True):
    """matches the laptop entries with df_GPUs and df_CPUs by add 2 foreign key columns.
    
    === Parameters:
    - df_laptops  : laptops data (after cleaning)
    - df_GPUs     : GPUs data (raw data)
    - df_CPUs     : CPUS data (raw data)
    
    === Returns: dataset with 2 extra columns 'cpu_name_key'and 'gpu_name_key'. """

    import difflib
    
    #     df_laptops['gpu_count'] = 0
    #     df_laptops['gpu_Name_key'] = "test"
    
    df_laptops['gpu_name'] = [name + (50 -len(name)) * " " for name in df_laptops.gpu_name]
    df_GPUs['Videocard Name'] = list([name + (50 -len(name)) * " " for name in df_GPUs['Videocard Name']])
    
    for idx_laptop, row_laptop in df_laptops.iterrows():  
        try:
            gpu_key = difflib.get_close_matches(df_laptops.loc[idx_laptop,'gpu_name'], df_GPUs['Videocard Name'] ,n=1)[0]
            df_laptops.loc[idx_laptop,'gpu_name_key'] = gpu_key
        except:
            df_laptops.loc[idx_laptop,'gpu_name_key'] = 'NOT FOUND'
        try:
            cpu_key = difflib.get_close_matches(df_laptops.loc[idx_laptop,'cpu_name'], df_CPUs['CPU Name'] ,n=1)[0]
            df_laptops.loc[idx_laptop,'cpu_name_key'] = cpu_key
        except:
            df_laptops.loc[idx_laptop,'cpu_name_key'] = 'NOT FOUND'
        
        if print_pairs:
            print("gpu_name :",df_laptops.loc[idx_laptop,'gpu_name'],'\n',
                  "gpu_key: ", gpu_key, '\n',
                  "cpu_name: ",df_laptops.loc[idx_laptop,'cpu_name'], '\n',
                  "cpu_key: ",cpu_key, '\n ===============================')
    
    df_laptops = df_laptops.drop(df_laptops[df_laptops['cpu_name_key'] == 'NOT FOUND'].index)
    df_laptops = df_laptops.drop(df_laptops[df_laptops['gpu_name_key'] == 'NOT FOUND'].index)
        
    return df_laptops

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

def get_performance(df_laptops,df_GPUs,df_CPUs,print_pairs = True):
    """adds cpu & gpu performance columns to the dataset.
      === Parameters:
    - df_laptops  : laptops data (after cleaning)
    - df_GPUs     : GPUs data (raw data)
    - df_CPUs     : CPUS data (raw data)
    
    === Returns: dataset with 5 extra performance columns (2 GPU, 3 CPU). """
    
    df_laptops['gpu_G3D_Mark'] = 0
    df_laptops['gpu_G2D_Mark'] = 0
    df_laptops['cpu_Thread_Mark'] = 0
    df_laptops['cpu_mark'] = 0
    df_laptops['cpu_cores'] = 0
    counter = 0
    df_GPUs['G3D Mark -']    = df_GPUs['G3D Mark -'].str.replace(",","")
    df_GPUs['G2D Mark -']    = df_GPUs['G2D Mark -'].str.replace(",","")
    df_CPUs['Cores']         = df_CPUs['Cores']
    df_CPUs['CPU Mark -']    = df_CPUs['CPU Mark -'].str.replace(",","")
    df_CPUs['Thread Mark -'] = df_CPUs['Thread Mark -'].str.replace(",","")

    for idx_laptop, row_laptop in df_laptops.iterrows():
        
        gpu_key = df_laptops.loc[idx_laptop,'gpu_name_key'] 
        df_laptops.loc[idx_laptop,'gpu_G3D_Mark'] = df_GPUs.query("{0} == @gpu_key".format('`Videocard Name`')).iloc[0]['G3D Mark -'].replace(",","")
        df_laptops.loc[idx_laptop,'gpu_G2D_Mark'] = df_GPUs.query("{0} == @gpu_key".format('`Videocard Name`')).iloc[0]['G2D Mark -'].replace(",","")
        
        cpu_key = df_laptops.loc[idx_laptop,'cpu_name_key'] 
        df_laptops.loc[idx_laptop,'cpu_Thread_Mark'] = df_CPUs.query("{0} == @cpu_key".format('`CPU Name`')).iloc[0]['Thread Mark -'].replace(",","")
        df_laptops.loc[idx_laptop,'cpu_mark'] = df_CPUs.query("{0} == @cpu_key".format('`CPU Name`')).iloc[0]['CPU Mark -'].replace(",","")
        df_laptops.loc[idx_laptop,'cpu_cores'] = df_CPUs.query("{0} == @cpu_key".format('`CPU Name`')).iloc[0]['Cores']
        
        if print_pairs:
            print(f'number {counter} done')
            counter += 1

    df_laptops['gpu_G3D_Mark'] = df_laptops['gpu_G3D_Mark'].astype(int)
    df_laptops['gpu_G2D_Mark'] = df_laptops['gpu_G2D_Mark'].astype(int)
    df_laptops['cpu_Thread_Mark'] = df_laptops['cpu_Thread_Mark'].astype(int)
    df_laptops['cpu_mark'] = df_laptops['cpu_mark'].astype(int)
    # df_laptops['cpu_cores'] = 0
            
    return df_laptops

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

def reconstruct_data(df):  
    import pandas as pd

    """Rebuilds One Hot Encoded columns back to full columns for analysis
    
    Parameter: Dataframe after complete preprocessing
    
    Returns: Dataframe without any One Hot Encoding"""
    
    unwanted = ['cpu_brand','cpu_cores','cpu_mark',
                'cpu_name','cpu_name_key','cpu_thread_mark',
                'gpu_brand','gpu_g2d_mark','gpu_g3d_mark',
                'gpu_name','gpu_name_key','inches','price_bracket',
                'price_euros','ram_gb','weight_kg','screen_resolution_l',
                'screen_resolution_w','memory_primary','memory_secondary']
    items = df.drop(unwanted ,axis =1).copy()
    # items.info()
    
    companies = items.loc[:,'company_acer':'company_toshiba'].copy()
    memories =  items.loc[:,
    'memory_flash_storage':'memory_ssd'].copy()
    
    opsystems = items.loc[:,'opsys_android':'opsys_windows 7'].copy()
    screens = items.loc[:,'screen_panel_ips':'screen_touchscreen'].copy()
    types = items.loc[:,'typename_2 in 1 convertible':'typename_ultrabook'].copy()
    columns = [companies,memories,opsystems,screens,types]

    def get_columns(row):
        keys = row.keys().to_list()
        column_name = keys[0].split("_")[0]
        row[column_name] = ""
        for key in keys:
            if row[key] == 1:
                row[column_name] +=  row[key] * " ".join(key.split("_")[1:]).capitalize() + ","
        if (row[column_name] == '') :
            row[column_name] = 'Other'  
        return row[column_name]
    
    names = ['companies','memories','opsystems','screens','types']
    dataframe = pd.DataFrame()
    for i in range(len(columns)):
        dataframe[names[i]] = columns[i].apply(get_columns,axis =1)
        if names[i] == 'types':
            dataframe[names[i]].replace({'Other':'Workstation'})
        elif names[i] == 'opsystems':
            dataframe[names[i]].replace({'Other':'No Os'})
        else:
            dataframe[names[i]].replace({'Other':'Other'})
            
    return pd.concat([df[unwanted],dataframe],axis = 1).sort_index(axis = 1)

# def Clean_data_pt1(df,y_train = False, calc_price_stats = False):
#     """Cleans the original 'laptop_price' dataset and restructes its columns.
    
#     === parameters ===
#     - df (pandas Dataframe): dataframe of laptop_prices
#     - y_train (Boolean) : dataframe 'df' includes a y_column 'Price_euros'.
    
#     = Returns : structed dataframe ready for 'get_CPU_performance'."""
    
    
#     import re
#     df = df.rename(columns={'Cpu':'cpu_name'})
    
#     #=== Processing Price brackets (if y_train = True) ==============================================
#     # calculate relevant statistics 
#     if y_train:
#         if calc_price_stats:
#             Price_std = df.Price_euros.std() * 0.7
#             Price_mean = df.Price_euros.mean()
#         else:
#             Price_std = 489.3
#             Price_mean = 1123.6
            
#         lower = Price_mean - Price_std
#         higher = Price_mean + Price_std

#         def laptop_price_range(row):
#             """creates 'Price_bracket' column (based on 'Price_euros'), a new category useful for future analysis."""
            
#             if row['Price_euros'] <= lower :
#                 row['Price_bracket'] = 'Budget'
#             elif (row['Price_euros'] > lower) & (row['Price_euros'] <= higher):
#                 row['Price_bracket'] = 'Mid Range'
#             else:
#                 row['Price_bracket'] = 'High End'
#             return row
        
#         df = df.apply(laptop_price_range,axis = 1)  
        
#     #=== Processing Storage =========================================================================
#     #- stroage types
#     df['Memory_SSD'] = (((df['Memory'].str.find('SSD') +1).values) > 0) #ex: returns bool if it finds 'SSD'
#     df['Memory_HDD'] = (((df['Memory'].str.find('HDD') +1).values) > 0)
#     df['Memory_Flash_Storage'] = (((df['Memory'].str.find('Flash Storage') +1).values) > 0)
#     df['Memory_Hybrid'] = (((df['Memory'].str.find('Hybrid') +1).values) > 0)
    
#     df[['Memory_SSD','Memory_HDD','Memory_Flash_Storage',
#         'Memory_Hybrid']]= df[['Memory_SSD','Memory_HDD','Memory_Flash_Storage',
#                                'Memory_Hybrid']].replace({True:1,False:0})  # maps {True:1,False:0} for ML model later
    
#     #- primary and secondary storage values 
#     memory = []
#     for sentence in df['Memory']:
#         #returns a list with memory sizes (ex: 250 SSD, 1TB HDD ==> [250,1])
#         items = [float(s) for s in re.findall(r'-?\d+\.?\d*', sentence)] 
#         memory.append(items)
#     memory = pd.DataFrame(memory,columns=['memory_Primary','memory_Secondary']) #creates dataframe_memory
    
#     #adjusts memory units (ex:1TB ==> 1000 GB)
#     memory['memory_Primary'] = memory['memory_Primary'].apply(lambda x: x*1000 if x<=13 else x)  
#     memory['memory_Secondary'] = memory['memory_Secondary'].apply(lambda x: x*1000 if x<=13 else 0)
    
#     df = pd.concat([df,memory], axis = 1) #merge new data into original dataset
    
#     #=== Processig Screen ==============================================================================
    
#     #- Screen_Resolution
#     def get_resolution(row):
#         """processes ScreenResolution into 3 columns."""
#         row['Screen_Resolution'] = row['ScreenResolution'].split(" ")[-1]
#         resolution_list = row['Screen_Resolution'].split("x")
#         row['Screen_Resolution_W'] = resolution_list[0] 
#         row['Screen_Resolution_L'] = resolution_list[1] 
#         return row
    
#     df = df.apply(get_resolution,axis = 1)
    
#     #Screen_Panel types (IPS,Touchscreen,Retina...etc.) into columns with Boolean values
#     df['Screen_Panel_IPS'] = (((df['ScreenResolution'].str.find('IPS') +1).values) > 0)
#     df['Screen_Touchscreen'] = (((df['ScreenResolution'].str.find('Touchscreen') +1).values) > 0)
#     df['Screen_Retina_Display'] = (((df['ScreenResolution'].str.find('Retina') +1).values) > 0)
    
#     #=== Processig Others (weight, Ram, OPSys and Type) ==================================================
#     #weight
#     df['Weight_kg'] = df['Weight'].str.replace("kg","")
#     df['Weight_kg'] = df['Weight_kg'].astype(float)

#     #Ram
#     df['Ram_GB'] = df['Ram'].str.replace("GB","")
#     df['Ram_GB'] = df['Ram_GB'].astype(int)
    
#     #reducing under represented/irrelevant categories
#     df['OpSys'] = df['OpSys'].replace({'Windows 10 S':'Windows 10','Mac OS X':'macOS'}) 
    
#     df['TypeName'] = df['TypeName'].replace({'Netbook':'Notebook'}) #'Netbooks' are a type of 'Notebooks'
    
#     #=== Processing Company =================================================================================
    
#     #reducing under represented/irrelevant categories
#     for other_1 in ['Mediacom','Microsoft','Razer','Microsoft','Google']:
#         df['Company'] = df['Company'].replace({other_1:'other (european/American)'})
#     for other_2 in ['Xiaomi','Chuwi','Huawei'] :
#         df['Company'] = df['Company'].replace({other_2:'other (Chinease/Korean)'})
#     for other_3 in ['Vero','Fujitsu']:
#         df['Company'] = df['Company'].replace({other_3:'other'})     
     
#     #=== Procesing CPU ======================================================================================
    
#     #cpu
#     def cpu_cleaning(row):
#         """processes df['cpu_name'] into multiple columns"""
        
#         cpu_specs = row['cpu_name'].split(" ")
#         row['cpu_brand'] = cpu_specs.pop(0) #returns the first item in the list cpu_specs
#         if "GHz" in cpu_specs[-1]:
#             row['cpu_clockspeed'] = cpu_specs.pop(-1)
            
#         # all 'Xeon' prcoessors are worded differently, needed custom handling
#         if 'Xeon' in cpu_specs:
#             row['cpu_model_name'] = cpu_specs.pop(0)
#             row['cpu_model_number'] = " ".join(cpu_specs)
#         else:
            
#             #extracting data based on observed patterns in 'cpu_name'
#             if (row['cpu_brand'] == 'Intel') & (
#             len(cpu_specs[-1]) - len([ch for ch in cpu_specs[-1] if ch.isdigit()]) in [1,2]) & (
#             len(cpu_specs[-1]) > 2): 
                
#                 row['cpu_model_number'] = cpu_specs.pop(-1)
                
#             elif (row['cpu_brand'] == 'Intel') & ("-" in cpu_specs[-1]):
                
#                 row['cpu_model_number'] = cpu_specs.pop(-1)
                
#             elif (row['cpu_brand'] == 'AMD'):
                
#                 row['cpu_model_number'] = cpu_specs.pop(-1)
                
#             row['cpu_model_name'] = " ".join(cpu_specs)
        
#         return row

#     df = df.apply(cpu_cleaning,axis =1)
    
#     #last minute adjustments
#     df['cpu_model_name'] = df['cpu_model_name'].str.replace("Celeron Dual Core","Celeron ")
    
#     #=== Processing GPU ======================================================================================
    
#     def gpu_cleaning_pt1(row):
#         """Processes Gpu column into multiple columns.
        
#         == Parameters:
#         - row : each row of the dataframe (for apply function)
        
#         === Returns: df with new columns:
#         - gpu_brand        : AMD,Nividia...etc.
#         - gpu_initial_name : gpu standard name
#         - gpu_model_number : ex GTX 1080 ==> 1080
#         - gpu_ending       : ex GTX 1080 Ti ==> 'Ti'
#         """
        
#         gpu_name = row['Gpu'].strip().split(" ")
#         row['gpu_brand'] = gpu_name.pop(0)
        
#         # #==> DELAYED DUE TO TIME CONSTRAINTS
#         #df['Gpu'] = df['Gpu'].str.replace("<U+039C>","",regex = False)
#         #df['gpu_model_number'] = np.nan
#         # has_digits = any(char.isdigit() for char in row['Gpu'])

#         # if has_digits:
#         #     s = [str(s) for s in re.findall(r'-?\d+\.?\d*', row['Gpu'])] #returns list of all avalible numbers
#         #     number = str(max(s,key=len)) 

#         #     for word in gpu_name:
#         #         if number in word:
#         #             row['gpu_model_number'] = word
#         #             number = word

#         #     last_word = gpu_name[-1]

#         #     if number in last_word:
#         #         gpu_name.remove(number)
#         #         row['gpu_initial_name'] = " ".join(gpu_name)
#         #     else:
#         #         gpu_desc = " ".join(gpu_name).split(number)
#         #         row['gpu_ending'] = gpu_desc.pop()
#         #         row['gpu_initial_name'] = gpu_desc.pop()
#         # else:
#         #     row['gpu_initial_name'] = " ".join(gpu_name)
            
#         return row
         
#     df = df.apply(gpu_cleaning_pt1, axis = 1)
#     df = df.rename(columns={'Gpu':'gpu_name'})
    
#     # #===> DELAYED DUE TO TIME CONSTRAINTS
#     #
#     # df = df.rename(columns={'gpu_initial_name':'gpu_line'})   

#     # lists_size = [len(my_list) for my_list in df.gpu_line.str.split(" ")]
#     # size = max(lists_size)

#     # print("Max size: ",size)

#     # gpu_line_df = pd.DataFrame(
#     # [item for item in df.gpu_line.str.split(" ").values],
#     # columns= [f'gpu_line_feature_{x}' for x in range(size)])

#     # df['gpu_line_feature_main'] = ""
#     # for i in range(size-1):
#     #     df['gpu_line_feature_main'] += gpu_line_df[f'gpu_line_feature_{i}'] + " " 

#     # df['gpu_line_feature_main'] = df['gpu_line_feature_main'].str.strip()

#     # df['gpu_line_feature_sec'] = gpu_line_df[f'gpu_line_feature_{size - 1}']
#     # df['gpu_line_feature_sec'] = df['gpu_line_feature_sec'].str.strip()
#     # indexes = df[df['gpu_line_feature_sec'] == ''][['gpu_line_feature_sec']].index
#     # df.loc[indexes,'gpu_line_feature_sec'] = np.nan

#     #Get_Dummies
#     cols = ['Company','TypeName','OpSys']
#     df = pd.get_dummies(df, columns=cols)

#     #drop extra
#     df = df.drop(['Screen_Resolution','ScreenResolution','Weight','Ram','Memory','Product','laptop_ID',
#                  'Company_other','TypeName_Workstation','OpSys_No OS'],axis = 1) #cpu_name
#     return df

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

# def get_CPU_performance(df_laptops, df_CPUs):
#     """Adds useful Columns describing the CPUs to dataframe and removes outliers 
#     that are hard to measure their performance.
    
#     === Parameters:
    
#     -df_laptops : original Dataset (after cleaning)
#     -df_CPUs    : CPUs dataset (raw) (source: https://www.cpubenchmark.net/CPU_mega_page.html )
    
#     === Returns: Laptop Dataset with extra Columns
    
#     -cpu_cores      : number of cores in CPU
#     -Cpu_Thread_Mark: Score of CPU's Threads (single Thread Performance)
#     -cpu_mark       : Score of CPU's Performance (all cores Performance)
#     """
    
#     df_laptops['Cpu_Thread_Mark'] = 0
#     df_laptops['cpu_mark'] = 0
#     df_laptops['cpu_cores'] = 0
#     df_laptops['Cpu_count'] = 0
#     df_laptops['cpu_Name_full'] = ''
    
#     for idx_laptop, row_laptop in df_laptops.iterrows():
        
#         number_isna = df_laptops.loc[idx_laptop,'cpu_model_number'] == np.nan
        
#         for idx_cpu, row_cpu in df_CPUs.iterrows():
            
#             cpu_name = df_CPUs.loc[idx_cpu,'CPU Name']
#             brand = df_laptops.loc[idx_laptop,'cpu_brand']
#             model_number = str(df_laptops.loc[idx_laptop,'cpu_model_number'])
#             model_name = df_laptops.loc[idx_laptop,'cpu_model_name']
            
#             if (
#                 (brand in cpu_name) & (model_number in cpu_name)
#                 ) | (
#                 (number_isna) & (brand in cpu_name) & (model_name in cpu_name)):
                
#                 thread_mark = int(df_CPUs.loc[idx_cpu,'Thread Mark -'].replace(",",""))
#                 cpu_mark = int(df_CPUs.loc[idx_cpu,'CPU Mark -'].replace(",",""))
#                 cpu_cores = df_CPUs.loc[idx_cpu,'Cores']
                
#                 df_laptops.loc[idx_laptop,'Cpu_Thread_Mark'] += thread_mark
#                 df_laptops.loc[idx_laptop,'cpu_mark'] += cpu_mark
#                 df_laptops.loc[idx_laptop,'cpu_cores'] += cpu_cores
#                 df_laptops.loc[idx_laptop,'cpu_Name_full'] = df_laptops.loc[idx_laptop,'cpu_Name_full'] + ',' + cpu_name
#                 df_laptops.loc[idx_laptop,'Cpu_count'] += 1
                
#         try:
#             if df_laptops.loc[idx_laptop,'Cpu_count'][i] > 1:
#                 df_laptops.loc[idx_laptop,'Cpu_Thread_Mark'] /= df_laptops.loc[idx_laptop,'Cpu_count']
#                 df_laptops.loc[idx_laptop,'cpu_mark'] /= df_laptops.loc[idx_laptop,'Cpu_count']
#                 df_laptops.loc[idx_laptop,'cpu_cores'] /= df_laptops.loc[idx_laptop,'Cpu_count']
                
#         except:
#             df_laptops.loc[idx_laptop,'cpu_final_performance'] = 0
            
#     #drop outliers
#     df_laptops = df_laptops[df_laptops['Cpu_count'] == 1]
#     df_laptops = df_laptops.drop('Cpu_count',axis = 1)
    
#     return df_laptops