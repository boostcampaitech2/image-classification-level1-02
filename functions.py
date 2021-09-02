import torch
from torchmetrics.functional import f1, accuracy
## Labeling Mask
def mapMask(file_path):
    if 'incorrect_mask' in file_path:
        return 'incorrect'
    elif 'normal' in file_path:
        return 'not wear'
    else:
        return 'wear'
    
def mapAgeGender(age, gender):
    answer = 0
    if age < 30:
        answer += 0
    elif age >= 60:
        answer += 2
    else:
        answer += 1
    return answer if gender == 'male' else answer + 3

## Labeling Mask+Age+Gender -> class
def mapping_class(mask, age, gender):
    answer = 0
    if mask == 'wear':
        answer += 0
    elif mask == 'incorrect':
        answer += 6
    else:
        answer += 12
        
    if gender == 'male':
        answer += 0
    else:
        answer += 3
    
    if age < 30:
        answer += 0
    elif age >= 60:
        answer += 2
    else:
        answer += 1
    
    return answer

def calculateAcc(y_pred, y_test, num_classes):    
    output = torch.argmax(y_pred, dim=1)
    return accuracy(output, y_test), f1(output, y_test, average='macro', num_classes=num_classes)