# Yous should change for the dataset you want to train on 
import os
import torch
import numpy as np
import math
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import get_custom_objects
import argparse

def pad(rst, length=1200, dim=1024):
  if len(rst) > length:
    return rst[:length]
  elif len(rst) < length:
    return np.concatenate((rst, np.zeros((length - len(rst), dim))))
  return rst

class generator:
  def __init__(self, embedding_file, dataframe):
    self.embedding_file = embedding_file
    self.dataframe = dataframe
  def __call__(self):
    with h5py.File(self.embedding_file, 'r') as embedding_matrix:
      for i in range(len(self.dataframe)):
        p1 = embedding_matrix[self.dataframe['p1'][i]]
        p1 = np.array(p1)
        x1= pad(p1)
        p2 = embedding_matrix[self.dataframe['p2'][i]]
        p2 = np.array(p2)
        x2= pad(p2)
        y = self.dataframe['label'][i]
        x1= tf.convert_to_tensor(x1,dtype=tf.float16)
        x2= tf.convert_to_tensor(x2,dtype=tf.float16)
        y= tf.convert_to_tensor(y,dtype=tf.float16)
        yield (x1,x2),y

def _fixup_shape(x):
    x1,x2,y = x
    x1.set_shape((seq_size, dim))
    x2.set_shape((seq_size, dim)) 
    y.set_shape(()) 

    return (x1, x2), y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "fasta_file", type=str,
        help = "fasta file"
    )
    parser.add_argument(
        "training_tsv", type=str,
        help = "train tsv file"
    )
    parser.add_argument(
        "test_tsv", type=str,
        help = "test file"
    )
    parser.add_argument(
        "validation_tsv", type=str,
        help = "validation file"
    )
    parser.add_argument(
        "checkpoint", type=str,
        help="A json file containing same info as the input file and alignments")
    parser.add_argument(
      "embedding_residue",type = str,
       help = "embedding residue string" 
     )
    parser.add_argument(
      "embedding_protein",type = str,
       help = "embedding protein string" 
     )
    parser.add_argument(
        "--epoch", type=int, default = 40,
        help = "epoch number"
    )
    parser.add_argument(
    "--batch_size",type = int, default=64,
    help = "batch size for loop" 
    )
  
    args = parser.parse_args()       
    seq_path =  args.fasta_file

    per_residue = True 
    per_residue_path = args.embedding_residue# where to store the embeddings

    # whether to retrieve per-protein embeddings 
    # --> only one 1024-d vector per protein, irrespective of its length
    per_protein = True
    per_protein_path = args.embedding_protein # where to store the embeddings

    # whether to retrieve secondary structure predictions
    # This can be replaced by your method after being trained on ProtT5 embeddings
    sec_struct = False
    sec_struct_path = "ss3_preds.fasta" # file for storing predictions

    # make sure that either per-residue or per-protein embeddings are stored
    assert per_protein is True or per_residue is True or sec_struct is True, print(
        "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")




    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))

   
    
   
  #OPTION 2 HDF5 stackoverflow get specific key instead of loading each of them

    if os.path.isfile(per_residue_path) == True:
      print("Already have the embedding file")

    

    ### Setting RAM GPU for training growth 
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    # Disables caching (when set to 1) or enables caching (when set to 0) for just-in-time-compilation. When disabled,
    # no binary code is added to or retrieved from the cache.
    os.environ['CUDA_CACHE_DISABLE'] = '0' # orig is 0

    # When set to 1, forces the device driver to ignore any binary code embedded in an application 
    # (see Application Compatibility) and to just-in-time compile embedded PTX code instead.
    # If a kernel does not have embedded PTX code, it will fail to load. This environment variable can be used to
    # validate that PTX code is embedded in an application and that its just-in-time compilation works as expected to guarantee application 
    # forward compatibility with future architectures.
    os.environ['CUDA_FORCE_PTX_JIT'] = '1'# no orig


    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT']='1'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"



    # =================================================
    mixed_precision.set_global_policy('mixed_float16')


    print("Load the pair dataset file")
    print(args.epoch)
    print(args.batch_size)
    #REWRITE COMPLETELY CURRENT TASK 16/05
    #LOAD AS BATCHES TRAIN WITH BATCHES
    train_dataframe = pd.read_csv(args.training_tsv, sep = '\t', header = None)
    train_array  = train_dataframe.to_numpy()
    train_dataframe = pd.DataFrame(train_array, columns = ['p1', 'p2', 'label'])
    train_dataframe['p1'] = train_dataframe['p1'].str.replace(".","_")
    train_dataframe['p2'] = train_dataframe['p2'].str.replace(".","_")

    validation_dataframe = pd.read_csv(args.validation_tsv, sep = '\t', header = None)
    validation_array  = validation_dataframe.to_numpy()
    validation_dataframe = pd.DataFrame(validation_array, columns = ['p1', 'p2', 'label'])
    validation_dataframe['p1'] = validation_dataframe['p1'].str.replace(".","_")
    validation_dataframe['p2'] = validation_dataframe['p2'].str.replace(".","_")

    test_dataframe = pd.read_csv(args.test_tsv, sep = '\t', header = None)
    test_array  = test_dataframe.to_numpy()
    test_dataframe = pd.DataFrame(test_array, columns = ['p1', 'p2', 'label'])
    test_dataframe['p1'] = test_dataframe['p1'].str.replace(".","_")
    test_dataframe['p2'] = test_dataframe['p2'].str.replace(".","_")


   
    seq_size = 1200
    dim = 1024
    total_size = len(train_dataframe)

    
    train_dataset = tf.data.Dataset.from_generator(
       generator(args.embedding_residue,train_dataframe),
       output_signature=((
          tf.TensorSpec(shape=(seq_size,dim),dtype=tf.float16),
          tf.TensorSpec(shape=(seq_size,dim),dtype=tf.float16)),
          tf.TensorSpec(shape=(),dtype=tf.float16)))
    train_dataset = train_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
  
    validation_dataset = tf.data.Dataset.from_generator(generator(args.embedding_residue,validation_dataframe),output_signature=((
          tf.TensorSpec(shape=(seq_size,dim),dtype=tf.float16),
          tf.TensorSpec(shape=(seq_size,dim),dtype=tf.float16)),
          tf.TensorSpec(shape=(),dtype=tf.float16)))
    validation_dataset = validation_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    
    test_dataset = tf.data.Dataset.from_generator(generator(args.embedding_residue,test_dataframe),output_signature=((
          tf.TensorSpec(shape=(seq_size,dim),dtype=tf.float16),
          tf.TensorSpec(shape=(seq_size,dim),dtype=tf.float16)),
          tf.TensorSpec(shape=(),dtype=tf.float16)))
    test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)


    def leaky_relu(x, alpha = .2):
      return tf.keras.backend.maximum(alpha*x, x)


    get_custom_objects().update({'leaky_relu': leaky_relu})
    seq_size = 1200
    dim = 1024
    def multi_cnn():
        DEPTH = 5
        WIDTH = 3
        POOLING_SIZE = 4
        FILTERS = 50
        KERNEL_SIZE = 2
        DEPTH_DENSE1 = 3
        DEPTH_DENSE2 = 2
        DROPOUT = DROPOUT1 = DROPOUT2 = 0.05
        DROPOUT_SPATIAL= 0.15
        ACTIVATION = 'swish'
        ACTIVATION_CNN = 'swish'
        INITIALIZER = 'glorot_normal'
        
        def BlockCNN_single(KERNEL_SIZE, POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2):
            c1 = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION_CNN, padding='same')
            x1 = c1(LAYER_IN1)
            x2 = c1(LAYER_IN2)

            g1 = Dropout(DROPOUT)(concatenate([GlobalMaxPooling1D()(x1),GlobalAveragePooling1D()(x1)]))
            a1 = GlobalAveragePooling1D()(x1)
            g2 = Dropout(DROPOUT)(concatenate([GlobalMaxPooling1D()(x2),GlobalAveragePooling1D()(x2)]))
            a2 = GlobalAveragePooling1D()(x1)

            x1 = SpatialDropout1D(DROPOUT_SPATIAL)(concatenate([MaxPooling1D(POOLING_SIZE)(x1), AveragePooling1D(POOLING_SIZE)(x1)]))
            x2 = SpatialDropout1D(DROPOUT_SPATIAL)(concatenate([MaxPooling1D(POOLING_SIZE)(x2), AveragePooling1D(POOLING_SIZE)(x2)]))

            return x1, x2, g1, g2, a1, a2

        def BlockCNN_multi(POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2, WIDTH):
          X1 = []
          X2 = []
          G1 = []
          G2 = []
          A1 = []
          A2 = []
          for i in range(2, 2+WIDTH):
            x1, x2, g1, g2, a1, a2 = BlockCNN_single(i, POOLING_SIZE, FILTERS, LAYER_IN1, LAYER_IN2)
            X1.append(x1)
            X2.append(x2)
            G1.append(g1)
            G2.append(g2)
            A1.append(a1)
            A2.append(a2)
          x1 = concatenate(X1)
          x2 = concatenate(X2)
          g1 = GlobalMaxPooling1D()(x1)
          g2 = GlobalMaxPooling1D()(x2)
          return x1, x2, g1, g2

        def BlockCNN_single_deep(KERNEL_SIZE, POOLING_SIZE, DEPTH, FILTERS, LAYER_IN1, LAYER_IN2):
          X1 = []
          X2 = []
          G1 = []
          G2 = []
          A1 = []
          A2 = []
          x1 = LAYER_IN1
          x2 = LAYER_IN2
          for i in range(DEPTH):
            x1, x2, g1, g2, a1, a2 = BlockCNN_single(KERNEL_SIZE, POOLING_SIZE, FILTERS, x1, x2)
            X1.append(x1)
            X2.append(x2)
            G1.append(g1)
            G2.append(g2)
            A1.append(a1)
            A2.append(a2)

          return X1, X2, G1, G2, A1, A2

        input1 = Input(shape=(seq_size, dim), name="seq1")
        input2 = Input(shape=(seq_size, dim), name="seq2")
        

        
        X1 = dict()
        X2 = dict()
        G1 = dict()
        G2 = dict()
        A1 = dict()
        A2 = dict()

        for i in range(KERNEL_SIZE, KERNEL_SIZE+WIDTH):
          X1[f'{i}'], X2[f'{i}'], G1[f'{i}'], G2[f'{i}'], A1[f'{i}'], A2[f'{i}'] = BlockCNN_single_deep(i, POOLING_SIZE, DEPTH, FILTERS, input1, input2)

        s1 = []
        s2 = []
        for i in range(KERNEL_SIZE, KERNEL_SIZE+WIDTH):
          s1.extend(G1[f'{i}'])
          s2.extend(G2[f'{i}'])

        s1 = concatenate(s1)
        s2 = concatenate(s2)
        
        s1 = BatchNormalization(momentum=.9)(s1)
        s2 = BatchNormalization(momentum=.9)(s2)

        s1 = Dropout(DROPOUT1)(s1)
        s2 = Dropout(DROPOUT1)(s2)
        
        s1_shape = s1.shape[-1]
        DENSE1 = 744 
        d1 = []
        for i in range(DEPTH_DENSE1):
            d1.append(Dense(int(DENSE1*(1/2)**i), kernel_initializer=INITIALIZER, activation=ACTIVATION))

        for i in range(DEPTH_DENSE1):
            s1 = d1[i](s1)
            s2 = d1[i](s2)
            s1 = Dropout(DROPOUT1)(s1)
            s2 = Dropout(DROPOUT1)(s2)
            
        s = concatenate([s1, s2])

        
        s_shape = s.shape[-1]
        DENSE2 = 328
            
        d2 = []
        for i in range(DEPTH_DENSE2):
            d2.append(Dense(int(DENSE2*(1/2)**i), kernel_initializer=INITIALIZER, activation=ACTIVATION))
    
        for i in range(DEPTH_DENSE2):
            s = d2[i](s)
            s = Dropout(DROPOUT2)(s)

        output = Dense(1, activation='sigmoid')(s)
        model = Model(inputs=[input1, input2], outputs=[output])
        
        adam = Adam(learning_rate=1e-3, amsgrad=True, epsilon=1e-6)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model


    model = multi_cnn()
    #model.summary()
    #tf.keras.utils.plot_model(model, show_shapes=True)

    checkpoint = args.checkpoint
    if os.path.isfile(checkpoint) == False:
      print("You did not pass the checkpoint file, so train from the beginning")
      # Run this when you need to train again
      early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5,        
        restore_best_weights=True  
    )
      # Define model checkpoint to save best model
      model_checkpoint = ModelCheckpoint(
          filepath = checkpoint,   # Save best model
          monitor='val_loss',
          save_best_only=True,
          
          )
      print("length of train dataframe: ",len(train_dataframe))
      print("validation of validation dataframe: ",len(validation_dataframe))
      # Train model with early stopping
      history = model.fit(
          train_dataset.repeat(),
          steps_per_epoch=math.ceil(len(train_dataframe)/args.batch_size),
          validation_data=validation_dataset.repeat(),
          validation_steps=math.ceil(len(validation_dataframe)/args.batch_size),
          epochs=args.epoch,  
          callbacks=[early_stopping, model_checkpoint])
    else:
      # Load the model with checkpoint file if you need to start with 
      model = tf.keras.models.load_model(checkpoint)

    y_pred = model.predict(test_dataset)
    y_pred = y_pred.flatten()
    y_true = test_dataframe['label'].values
    y_true = y_true.astype(int)
    
    cm1=confusion_matrix(y_true, np.round(y_pred).astype(int))
    acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
    spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
    prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
    f1 = 2 * (prec * rec) / (prec + rec)
    mcc = matthews_corrcoef(y_true,np.round(y_pred).astype(int))

    prc = metrics.average_precision_score(y_true, y_pred)

    print("============= INFERENCE BY NEURAL NETWORK ===============")
    try:
      auc = metrics.roc_auc_score(y_true, y_pred)
      print(f'accuracy: {acc}, precision: {prec}, recall: {rec}, specificity: {spec}, f1-score: {f1}, mcc: {mcc}, auroc: {auc}, auprc: {prc} ')
      print(str(acc) + "\t" + str(prec) + "\t" + str(rec) + "\t" + str(spec) + "\t" + str(f1) + "\t" + str(mcc)+"\t" + str(auc)  + "\t" + str(prc) + "\n")
    except ValueError:
      print(f'accuracy: {acc}, precision: {prec}, recall: {rec}, specificity: {spec}, f1-score: {f1}, mcc: {mcc}, auroc: nan, auprc: {prc} ')
      print(str(acc) + "\t" + str(prec) + "\t" + str(rec) + "\t" + str(spec) + "\t" + str(f1) + "\t" + str(mcc)+"\t nan"  + "\t" + str(prc) + "\n")



