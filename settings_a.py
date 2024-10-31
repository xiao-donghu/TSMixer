# @Time         ：2024/9/13 15:13
# @Author       : 胡晓东
# @Email        : @.com
# @File         : settings_a.py
# @Description  :


# 数据库表对接
input_database_table_a = {
    'database': 'flow_jsc',
    'user': 'root',
    'password': 'Yuce@2020pwd',
    'host': '41.248.129.109',
    'port': '3306',
    'table': 'zs_kk_minute_stat_ft',
}

## 输出表，用于上屏
out_database_table_a = {
    'database': 'flow_jsc',
    'user': 'root',
    'password': 'Yuce@2020pwd',
    'host': '41.248.129.109',
    'port': '3306',
    'table': 'zs_zdxz_actual_prediction_a',
}

## 输出表1，用于长期监控模型性能
out_database_table_a_1 ={
    '待lk创建'




}

# 字段映射

input_table_schema_a = {
    'kk_code':'kk_code',
    'kk_name':'kk_name',
    'kk_type':'kk_type',
    'start_time':'window_start',
    'end_time':'window_end',
    'flow':'kk_flow',
    'create_time': 'create_time',
    'modify_time':'modify_time',
}

output_table_schema_a = {
    'date_range': 'date_range', #预测的日期范围，三类，节假日从第一天到最后一天，周末第一天到第二天，工作日当日日期
    'time': 'time',
    'date_type': 'date_type',  #日期类型，三类，如‘2024年中秋节’’周末‘’工作日‘
    'year': 'year',  # 年份，如2024
    'location': 'kk_name',    #如金塘
    'y_pred_in': 'in_flow_pre',
    'y_pred_out': 'out_flow_pre',
    'in_high_time': 'in_hour_pre', #数据汇总到小时粒度，然后获取一天中流量最大的时间段，每天都填同一值
    'out_high_time': 'out_hour_pre',
    'record_time':'record_time',
}

output_table_schema_a_1 = {
    'date': 'date',
    'time': 'time',
    'date_type': 'date_type',  #日期类型，三类，如‘2024年中秋节’’周末‘’工作日‘
    'location': 'location',
    'direction': 'direction',
    'model_type': 'model_type',
    'model_name': 'model_name',
    'y_pred': 'pred_flow',
}
