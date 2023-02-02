import json


def json_label_del(path, del_label_list):    
    with open(path, 'r', encoding='utf-8') as input_f :
        data = json.load(input_f)

    annos = data['annotations']
    for i in range(len(annos) - 1, -1, -1) :
        del_check = annos[i]['object_class'] in del_label_list
        if del_check :
            del annos[i]
    
    category = data['category']
    for i in range(len(category) -1, -1, -1) :
        del_check = category[i]['object_class_name'] in del_label_list
        if del_check :
            del category[i]
    
    print(json.dumps(data, ensure_ascii=False, indent=3))
    
    with open(path, 'w', encoding='utf-8') as output_f :
    json.dump(data, output_f, indent=4)
