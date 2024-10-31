# @Time         ：2024/1/24 15:13
# @Author       : 丁连涛
# @Email        : dingliantao@yucekj.com
# @File         : settings_a.py
# @Description  :


# todo 修改输入表、输出表、输入表schema、输出表schema
input_database_table_b = {
    'database': 'flow_jsc',
    'user': 'root',
    'password': 'Yuce@2020pwd',
    'host': '41.248.129.109',
    'port': '3306',
    'table': 'zs_kk_minute_sta_ft1',
}

out_database_table_b = {
    'database': 'flow_jsc',
    'user': 'root',
    'password': 'Yuce@2020pwd',
    'host': '41.248.129.109',
    'port': '3306',
    'table': 'zs_zdxz_actual_prediction_b',
}

## 输出表1，用于长期监控模型性能
out_database_table_b_1 ={
    '待lk创建'




}


input_table_schema_b = {
    'kk_code':'kk_code',
    'kk_name':'kk_name',
    'kk_type':'kk_type',
    'start_time':'window_start',
    'end_time':'window_end',
    'flow':'kk_flow',
    'create_time': 'create_time',
    'modify_time':'modify_time',
}


output_table_schema_b = {
    'date_range': 'date_range', #预测的日期范围，三类，节假日从第一天到最后一天，周末第一天到第二天，工作日当日日期
    'year':'year',
    'date_type': 'date_type',  #日期类型，三类，如‘2024年中秋节’’周末‘’工作日‘
    'location': 'kk_name',    #如金塘
    'start_time':'window_stat', #预测时间的起点
    'end_time':'window_end',     #预测时间的终点
    'pred_time':'time_type',   #预测时段类型，如'30分钟''六十分钟'
    'y_pred_in': 'in_flow_pre',
    'y_pred_out': 'out_flow_pre',
    'record_time':'record_time',


}

output_table_schema_b_1 = {
    'date': 'date',
    'time': 'time',
    'data_type': 'data_type',  #日期类型，三类，如‘2024年中秋节’’周末‘’工作日‘
    'location': 'location',
    'direction': 'direction',
    'model_type': 'model_type',
    'model_name': 'model_name',
    'y_pred': 'pred_flow',
}


location_list = ['舟山西', '朱家尖']

direction_list = ['in', 'out']