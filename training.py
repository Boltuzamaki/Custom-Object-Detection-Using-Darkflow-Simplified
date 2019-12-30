from darkflow.net.build import TFNet              #darkflow is supporting library to build model     

options = {'model' : 'cfg/yolo.cfg',              #making dictionary for inputs to feed in model
               'load' : 'bin/yolo.weights',
               'batch' : 32,
               'epoch' : 100,
               'gpu' : 0.5,
               'train' : True,
               'annotation' : './xml',
               'dataset' : './Image',
               'threshold' : 0.1
              }

tfnet = TFNet(options)                            #setting up layers
tfnet.train()                                     #train