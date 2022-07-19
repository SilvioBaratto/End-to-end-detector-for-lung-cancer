import datetime


import os
import shutil

from util.util import importstr
from util.logconf import logging
log = logging.getLogger('nb')

def run(app, *argv):
    argv = list(argv)
    argv.insert(0, '--num-workers=48')  # <1>
    log.info("Running: {}({!r}).main()".format(app, argv))
    
    app_cls = importstr(*app.rsplit('.', 1))  # <2>
    app_cls(argv).main()
    
    log.info("Finished: {}.{!r}).main()".format(app, argv))

# clean up any old data that might be around.
# We don't call this by default because it's destructive, 
# and would waste a lot of time if it ran when nothing 
# on the application side had changed.
def cleanCache():
    shutil.rmtree('data-unversioned/cache')
    os.mkdir('data-unversioned/cache')

# cleanCache()

training_epochs = 20
experiment_epochs = 10
final_epochs = 50

training_epochs = 2
experiment_epochs = 2
final_epochs = 5
seg_epochs = 10

if __name__ == '__main__':
    run('prepcache.LunaPrepCacheApp')
    run('training.ClassificationTrainingApp', f'--epochs=100', 'nodule-nonnodule')
    run('training.ClassificationTrainingApp', f'--epochs=40', '--malignant', '--dataset=MalignantLunaDataset',
    '--finetune=''../LUNA/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
    'finetune-head')
    run('training.ClassificationTrainingApp', f'--epochs=40', '--malignant', '--dataset=MalignantLunaDataset',
    '--finetune=''../LUNA/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
    '--finetune-depth=2',
    'finetune-depth2')
    run('nodule_analysis.NoduleAnalysisApp', '--run-validation')
    run('nodule_analysis.NoduleAnalysisApp', '--run-validation', '--malignancy-path')
    
    
    
    
