## Labeling Mask
def mask(file_path):
    if 'incorrect_mask' in file_path:
        return 'incorrect'
    elif 'normal' in file_path:
        return 'not wear'
    else:
        return 'wear'

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

