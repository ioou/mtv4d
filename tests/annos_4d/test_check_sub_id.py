def test_sub_id():        
    data1 = '11000011000'
    data2 = '00001100011'
    data3 = '11111111111'
    acc_thres = 2
    output = len(data1) * [0]
    st_flag = True
    for ts, visible in enumerate(data1):
        if not visible:
            acc_ts_list += [ts]
        else:
            if len(acc_ts_list) >= acc_thres and not st_flag:
                sub_id +=1
            else:  # 如果没超过，之前的要补上
                for j in acc_ts_list: 
                    output[j] = sub_id
            st_flag = False
    
