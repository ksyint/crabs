import os
import json

eval_sample_list = [('./output/test6_7_model001_thres99_tmppanel', ['tmp_panel9']),
                    ('./output/test6_7_model003_thres99_tmppanel', ['tmp_panel9']),
                    ('./output/test6_6_model001_thres99_tmppanel', ['tmp_panel8']),
                    ('./output/test6_6_model003_thres99_tmppanel', ['tmp_panel8']),
                    ('./output/test6_5_model001_thres99_tmppanel', ['tmp_panel6','tmp_panel7']),
                    ('./output/test6_5_model003_thres99_tmppanel', ['tmp_panel6','tmp_panel7']),
                    ('./output/test6_4_model001_thres99_tmppanel', ['tmp_panel5']),
                    ('./output/test6_4_model003_thres99_tmppanel', ['tmp_panel5']),
                    ]

for result_path, labels in eval_sample_list:
    json_result_path = os.path.join(result_path, 'json_results')
    json_namelist = sorted(os.listdir(json_result_path))
    total_data = len(json_namelist)
    num_accurate = 0

    for json_name in json_namelist:
        json_path = os.path.join(json_result_path, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data['labels'][0] in labels:
            num_accurate +=1

    accuracy = num_accurate / total_data
    save_path = os.path.join(result_path, 'eval.txt')
    with open(save_path, 'w') as f:
        f.write(str(accuracy))
