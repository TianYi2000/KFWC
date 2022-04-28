import os

models = [
    'resnet18',
    # 'resnet34',
    # 'resnet50',
    # 'resnest50',
    # 'scnet50',
    # 'inceptionv3',
    'vgg16',
    # 'vgg19'
]

phases = [
    'single_image',
    'single_complaint',
    'single_lesion',
    'single_lesion_complaint'
]


for model in models:
    for phase in phases:
        for lock in [False, True]:
            if phase == 'single_image' and lock:
                continue
            elif phase == 'single_image':
                print(f'{model} {phase}:')
            else:
                print(f'{model} {phase} lock={lock}:')
            path = os.path.join('model', phase)
            filenames=os.listdir(path)
            match_models = []
            for filename in filenames:
                if model in filename:
                    if phase == 'single_image':
                        match_models.append(filename)
                    else:
                        if (lock and 'lock' in filename) or (not lock and 'lock' not in filename):
                            match_models.append(filename)

            match_models.sort(reverse=True)
            model_path = os.path.join(path, match_models[0])
            assert (model_path)
            os.system(f'python test_{phase}.py --use_gpu 1 --two_stream_path {model_path} --batch_size 4 --oct_size {299 if model == "inceptionv3" else 224}')