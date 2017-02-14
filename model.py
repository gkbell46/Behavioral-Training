from utilityFile import readcsvfile, data_generator_correction, resize_images, augument, gen_training_data, run_save_model


if __name__ == '__main__':
    print("model imported","Reading data")
    lines = readcsvfile() 
    print("generating data")
    images , measurements = data_generator_correction(lines)
    print("resizing images")
    images = resize_images(images)
    print("augumenting images")
    augumented_images, augumented_measurements = augument(images, measurements)
    print("Generate Taining data")
    X_train, y_train = gen_training_data(augumented_images,augumented_measurements)
    nb_epochs = 5

    print(len(lines))
    print(len(images))
    print(len(measurements))
    print("Run model")
    run_save_model(X_train,y_train,nb_epochs)
    print("Data generated")