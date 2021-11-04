import pandas as pd
import math

def get_avg_df(_df, num_avg_days=7, num_days_before=3):
    '''
    input STATION-WISE dataframe with all expected columns
    returns dataframe with max_waterlevel and avg_waterlevel columns unchanged 
        and average of 'num_avg_days' number of days worth other features 
        and starting from 'num_days_before' ago
        
    example: num_avg_days=7, num_days_before=3
        then row for January 10 will have max and avg waterlevel data of Jan 10 
            and other columns will have average of values from Jan 1 to 7
    '''
    df=_df.copy()
    
    MONTH_COL = 'Month'
    MAX_TEMP_COL = 'Max Temp. (degree Celcius)'
    MIN_TEMP_COL = 'Min Temp. (degree Celcius)'
    RAINFALL_COL = 'Rainfall (mm)'
    ACTUAL_EVA_COL = 'Actual Evaporation (mm)'
    REL_HUMIDITY_M_COL = 'Relative Humidity (morning, %)'
    REL_HUMIDITY_A_COL = 'Relative Humidity (afternoon, %)'
    SUNSHINE_COL = 'Sunshine (hour/day)'
    CLOUDY_COL = 'Cloudy (hour/day)'
    SOLAR_RAD_COL = 'Solar Radiation (cal/cm^2/day)'
    MIN_WATERLEVEL = 'MIN_WL(m)'
    MAX_WATERLEVEL = 'MAX_WL(m)'
    AVG_WATERLEVEL = 'AVE_WL(m)'

    months, min_temps, max_temps, rainfalls, actual_evas, rhs_m, rhs_a, sunshines, cloudies, solar_rads = \
    [], [], [], [], [], [], [], [], [], [] 
    max_waterlevels, min_waterlevels = [], []

    # populate list with daily features
    months = [x for x in df[MONTH_COL]]
    min_temps = [x for x in df[MIN_TEMP_COL]]
    max_temps = [x for x in df[MAX_TEMP_COL]]
    rainfalls = [x for x in df[RAINFALL_COL]]
    actual_evas = [x for x in df[ACTUAL_EVA_COL]]
    rhs_m = [x for x in df[REL_HUMIDITY_M_COL]]
    rhs_a = [x for x in df[REL_HUMIDITY_A_COL]]
    sunshines = [x for x in df[SUNSHINE_COL]]
    cloudies = [x for x in df[CLOUDY_COL]]
    solar_rads = [x for x in df[SOLAR_RAD_COL]]
    max_waterlevels = [x for x  in df[MAX_WATERLEVEL]]
    avg_waterlevels = [x for x  in df[AVG_WATERLEVEL]]

    def get_avg_in_range(vals, start, end):
        '''
        returns average of list values from start to end index 
        '''
        total = 0.0
        cnt = 0
        for i in range(start, end+1):
            if math.isnan(vals[i]):
                continue
            total+=vals[i]
            cnt+=1
            
        if cnt==0: 
            return math.nan
        return float(total/cnt);

    new_months, new_min_temps, new_max_temps, new_rainfalls, new_actual_evas, \
    new_rhs_m, new_rhs_a, new_sunshines, new_cloudies, new_solar_rads = [], [], [], [], [], [], [], [], [], [] 

    output_avg_waterlevels, output_max_waterlevels = [], []
    
    # populate new features with previous average values
    for curr_idx in range(num_avg_days+num_days_before, df.shape[0]):
        avg_start_idx = curr_idx-(num_avg_days+num_days_before)
        avg_end_idx = avg_start_idx+num_days_before-1
        
        new_min_temps.append(get_avg_in_range(min_temps, avg_start_idx, avg_end_idx))
        new_max_temps.append(get_avg_in_range(max_temps, avg_start_idx, avg_end_idx))
        new_actual_evas.append(get_avg_in_range(actual_evas, avg_start_idx, avg_end_idx))
        new_rhs_m.append(get_avg_in_range(rhs_m, avg_start_idx, avg_end_idx))
        new_rhs_a.append(get_avg_in_range(rhs_a, avg_start_idx, avg_end_idx))
        new_sunshines.append(get_avg_in_range(sunshines, avg_start_idx, avg_end_idx))
        new_cloudies.append(get_avg_in_range(cloudies, avg_start_idx, avg_end_idx))
        new_solar_rads.append(get_avg_in_range(solar_rads, avg_start_idx, avg_end_idx))
        new_rainfalls.append(get_avg_in_range(rainfalls, avg_start_idx, avg_end_idx))
        
        # in case days fall in two months, set the month that covers most days
        new_months.append(int(get_avg_in_range(months, avg_start_idx, avg_end_idx)))
        
        output_max_waterlevels.append(max_waterlevels[curr_idx])
        output_avg_waterlevels.append(avg_waterlevels[curr_idx])

    return pd.DataFrame({MONTH_COL: new_months,
                         'Avg '+ MAX_TEMP_COL: new_max_temps,
                         'Avg '+ MIN_TEMP_COL: new_min_temps,
                         'Avg '+ RAINFALL_COL: new_rainfalls,
                         'Avg '+ ACTUAL_EVA_COL: new_actual_evas, 
                         'Avg '+ REL_HUMIDITY_M_COL: new_rhs_m,
                         'Avg '+ REL_HUMIDITY_A_COL: new_rhs_a,
                         'Avg '+ SUNSHINE_COL: new_sunshines,
                         'Avg '+ CLOUDY_COL: new_cloudies,
                         'Avg '+ SOLAR_RAD_COL: new_solar_rads,
                         MAX_WATERLEVEL: output_max_waterlevels,
                         AVG_WATERLEVEL: output_avg_waterlevels
                        })
