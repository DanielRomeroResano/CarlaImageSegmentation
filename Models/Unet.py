from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Input, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



def encoderBlock(inputs, filters, maxPool=True):
    
    X = Conv2D(filters, 3, padding="same")(inputs)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = Conv2D(filters, 3, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    skip = X
    
    if maxPool:
        X = MaxPooling2D(pool_size=(2, 2))(X)        

    return X, skip



def decoderBlock(inputs, skip, filters):
    
    X = Conv2DTranspose(filters, 3, strides=(2,2), padding="same")(inputs) 
    
    X = concatenate([X, skip], axis=3) #concat prev skip connection

    X = Conv2D(filters, 3, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
    X = Conv2D(filters, 3, padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    
  
    return X


def unet(inputShape,labels, filters):
    
    inputs = Input(inputShape)
        
    # Encoder
    X1, S1 = encoderBlock(inputs, filters, maxPool=True)
    X2, S2 = encoderBlock(X1, filters * 2, maxPool=True)
    X3, S3 = encoderBlock(X2, filters * 4, maxPool=True)
    X4, S4 = encoderBlock(X3, filters * 8, maxPool=True)
    
    X5, S5 = encoderBlock(X4, filters * 16, maxPool=False)
    
    # Decoder
    X6 = decoderBlock(X5, S4, filters * 8)
    X7 = decoderBlock(X6, S3,  filters * 4)
    X8 = decoderBlock(X7, S2,  filters = filters * 2)
    X9 = decoderBlock(X8, S1,  filters = filters)

    X10 = Conv2D(filters, 3,activation='relu', padding='same')(X9)

 
    X11 = Conv2D(filters = labels, kernel_size = (1,1), activation='sigmoid', padding='same')(X10)
    
    modelUnet = Model(inputs=inputs, outputs=X11)

    return modelUnet