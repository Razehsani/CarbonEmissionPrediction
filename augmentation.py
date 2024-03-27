import pandas as pd
import numpy as np

dataset = pd.read_csv(r"C:\Users\m.rasooli\Desktop\Razieh\Potato\Augmentation\lentils2.csv", encoding="utf8")
consts = [
    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],#Cultivated area (hectares)
    [1, 2],#خاک ورز مرکب (Frequency)
    [1, 2],#سایر عملیات(Frequency)
    (85, 130),#بذرو ضدعفونی(کیلوگرم)
    [1, 2, 3],#بذر کاری(Frequency)
    (90, 125),#کود شیمیایی اوره(کیلوگرم)
    (40, 68),#کود شیمیایی فسفات(کیلوگرم)
    (0, 5),#کود شیمیایی پتاس(کیلوگرم)
    (0, 15),#ریزمغذی ها
    (200, 287),#حمل کود و بذر(کیلوگرم)
    [1, 2, 3],#کودپاشی(Frequency)
    (0.5, 1.43),#حشره کش(لیتر)
    (1.5, 2.8),#سم علف کش(لیتر)
    [1, 2],#سمپاشی(Frequency)
    [1, 2],#cultivator(Frequency)
    (4000, 5600),#آب مصرفی(مترمکعب)
    [2, 3, 4],#آبیاری(Frequency)
    [1, 2],#برداشت و جمع آوری((Frequency))
    (1600, 2150),#بارگیری و تخلیه(کیلوگرم)
    (1600, 2150)#عملکرد در هکتار
]

augmented_list = []
augmented_list.append(list(dataset.values[0]))

for i in range(1000):
    augmented_row = [];
    cultivated_area = 0;
    for j in range(len(consts)):
        if(type(consts[j]) is tuple):
            if(j == 19):
                augmented_row.append(augmented_row[18])
            else:
                value = np.random.uniform(consts[j][0], consts[j][1]) * cultivated_area
                
                if(consts[j][1] <= 5):
                    value = round(value, 1)
                else:
                    value = round(value)
                
                augmented_row.append(value)
        elif (type(consts[j]) is list):
            if(j == 0):
                cultivated_area = np.random.choice(consts[j])
                augmented_row.append(cultivated_area)                
            else:
                augmented_row.append(np.random.choice(consts[j]))
    
    augmented_row.append(np.nan)
    augmented_row.append(np.nan)
    augmented_list.append(augmented_row)
    
df = pd.DataFrame(augmented_list, columns=dataset.columns)
df.to_csv(r"C:\Users\m.rasooli\Desktop\Razieh\Potato\Augmentation\lentils2_aug.csv", encoding="utf8")