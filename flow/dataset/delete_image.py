import os

labels = ["DDoS", "FTP-Patator", "normal", "PortScan", "SSH-Patator", "web_scan_dir"]
root_path = '../../../../datasets/CIC-IDS2017/MachineLearningCVE/real/defense_dataset2/real'
save_number = 500

for label in labels:
    real_path = root_path + '/' + label
    file_list = os.listdir(real_path)
    if len(file_list) < save_number:
        continue
    remove_list = file_list[save_number:-1]
    for file_path in remove_list:
        remove_path = real_path + '/' + file_path
        os.remove(remove_path)
        print(f'remove {remove_path}')
