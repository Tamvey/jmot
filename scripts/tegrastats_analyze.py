import os

def get_ram_from_str(str):
    i1 = str.find("/")
    return int(str[:i1])

def get_metrics(data):
    if not data:
        return 
    
    temp = []
    out = {}
    
    for i in data: 
        if i != 0: temp.append(i)
    temp.sort()

    n = len(temp)   
    mean = sum(temp) / n
    variance = sum((x - mean) ** 2 for x in temp) / n

    out["mean"] = mean
    out["P0.95"] = temp[int(n * 0.95)]
    out["P0.99"] = temp[int(n * 0.99)]
    out["stdev"] = variance ** 0.5
    out["usage"] = temp[-1] - temp[0]
    return out 


def read_file_lines(filepath):
    out = []
    with open(filepath) as fr:
        while True:
            line = fr.readline()
            if not line: 
                break
            out.append(line.split())
    return out 

def compute_stats(data):
    out = {}

    # GPU usage
    index_gpu_usage = data[0].index("GR3D_FREQ") + 1
    all_gpu_usage = []
    # if file broken
    for i in data:
        try: 
            val = int(i[index_gpu_usage][:-1]) 
            all_gpu_usage.append(val)
        except Exception: 
            pass 
    print("gpu len: ", len(all_gpu_usage))
    gpu = get_metrics(all_gpu_usage)
    out["gpu"] = gpu

    # RAM
    index_ram_util = data[0].index("RAM") + 1
    all_ram_util = []
    # if file broken
    for i in data:
        try: 
            val = get_ram_from_str(i[index_ram_util][:-1]) 
            all_ram_util.append(val)
        except Exception: 
            pass 
    print("ram len: ", len(all_ram_util))
    ram = get_metrics(all_ram_util)
    out["ram"] = ram

    return out 

def print_to_csv(res, model_name, file_name):
    with open(file_name, "a") as fw:
        data = [model_name, 
                res["gpu"]["mean"], res["gpu"]["P0.95"], res["gpu"]["P0.99"], res["gpu"]["stdev"], 
                res["ram"]["usage"]
                ]
        temp = list(map(lambda x: str(x), data))
        fw.write(",".join(temp) + "\n")

def main(root_path):
    root_content = os.listdir(root_path)
    out_csv = "out_onnx.csv"
    for filename in root_content:
        # begins with t_
        if (not filename.startswith("t_")):
            continue 

        model_name = filename[2:-4]
        print(model_name)

        # read measures from file
        data = read_file_lines(root_path + "/" + filename) # ugly path usage

        # computations
        res = compute_stats(data)

        # reflect in csv
        print_to_csv(res, model_name, out_csv)

if __name__ == "__main__":
    main("/home/matvey/diploma/timings/onnx")