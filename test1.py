
import json

def main():
    f = open('./annotations_gt.json')
    data = json.load(f)
    classes = ['hug', 'shake hands', 'kiss', 'fencing', 'arm wrestling', 'cut hair']
    files = []
    fl = open('./fineactionvids.csv', 'w')

    for val in data['database'].values():
        fileName = val['filename']
        for ann_val in val['annotations']:
            lab = ann_val['label']
            if lab in classes:
                files.append([fileName, lab])
                fl.write(fileName + ', ' + lab + '\n')
                break

    f.close()
    fl.close()


if __name__ == '__main__':
    main()
