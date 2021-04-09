# %%
import re
import csv
# %%
f = open('./data/chatmem.txt', 'r')
f_pre = open('./data/chatmem_pre.tsv', 'w', encoding='utf-8')
tsv_w = csv.writer(f_pre, delimiter='\t')
lines = f.readlines()
flag = 0
for i in range(len(lines)):
    line = lines[i]
    if flag == 0: tmp_buffer = ''
    if line:
        if re.match(r'\b(\d\d\d\d-\d\d-\d\d+.*)', line) == None: # 匹配日期行
            # print("1---" + line.strip('\n').strip() + "---2")
            if line.strip('\n').strip() != '':
                # print(line)
                if flag == 0:
                    tmp_buffer = line.strip('\n')
                else: 
                    # tmp_buffer += line
                    tsv_w.writerow([tmp_buffer, line.strip()])
                flag = 1 if flag == 0 else 0
                # print(tmp_line)
f.close()
f_pre.close()

# %%

# %%
