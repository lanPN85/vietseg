import os

from learn import *


if __name__ == '__main__':
    net = network.Network([SIZE, HIDDEN, 3])

    names = os.listdir('dat/iob')
    for i in range(len(names)):
        names[i] = os.path.join('dat/iob', names[i])
    training_data, test_data = get_full_train_test(names)
    acc = net.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA,
                  lmbda=LAMBDA,
                  evaluation_data=test_data,
                  monitor_evaluation_accuracy=True)
    net.save('../var/params.net'.
             format(HIDDEN, EPOCHS, MINI_BATCH_SIZE, ETA, LAMBDA, WINDOW, SHAPE, RUN))
    print('Accuracy: %.2f' % acc)
