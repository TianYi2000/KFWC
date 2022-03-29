import csv
import time, datetime

info_file = 'info-2.csv'
root_file = 'D:\\Projects\\AMD_processed\\label\\new_two_stream\\OCT\\all_label.csv'
dest_file = 'OCT_label+info.csv'

info = dict()
with open(info_file, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    data = data[1:]
    for line in data:
        name = line[0]
        timeArray = time.strptime(line[3], "%Y/%m/%d")
        timeStamp = time.mktime(timeArray)
        if name not in info:
            info[name] = dict()
        info[name][timeStamp] = line[2]
print(info)

with open(root_file, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    title = data[0]
    data = data[1:]
title.append('information')

with open(dest_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(title)
    for line in data:
        url = line[0]
        date = url.split('/')[-1][:10]
        pos = line[10]
        if '无法区分' in pos:
            pos = '无法区分'
        if date[:2] == '20':
            timeArray = time.strptime(date, "%Y-%m-%d")
            timeStamp = time.mktime(timeArray)
            name = url.split('/')[1]

            right_date = None
            difference = float("inf")
            if name in info:
                for key in info[name].keys():
                    if difference > abs(key - timeStamp) and pos in info[name][key] or '双眼' in info[name][key] :
                        difference = abs(key - timeStamp)
                        right_date = key
                if right_date != None:
                    line.append(info[name][right_date])
                # dateArray = datetime.datetime.utcfromtimestamp(right_date)
                # otherStyleTime = dateArray.strftime("%Y-%m-%d")
                # line.append(otherStyleTime)
                # line.append(name)
            writer.writerow(line)
        else:
            writer.writerow(line)

