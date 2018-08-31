import numpy
from PIL import Image
from resizeimage import resizeimage
import matplotlib.image as img
import os
import random
from terminaltables import AsciiTable

def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))

def dsigmoid(x):
    return x * (1.0 - x)


def compute_result(input_sample):
    l1 = sigmoid(numpy.dot(input_sample, first_layer))
    l2 = sigmoid(numpy.dot(l1, second_layer))
    maximum = 0
    selected_index = 0
    for index in range(10):
        if l2[index] > maximum:
            maximum = l2[index]
            selected_index = index

    return selected_index


def print_sample(input_sample):
    input_sample = input_sample.reshape(16, 16).tolist()
    text = []
    for sample_row in range(16):
        text_row = input_sample[sample_row]
        text_row = map(lambda cell: '*' if cell == 1 else ' ', text_row)
        text_row = ''.join(text_row)
        text.append(text_row)
    return '\n'.join(text)


numpy.set_printoptions(threshold='nan', suppress=True)

samples = numpy.empty([0, 256])
results = numpy.empty([0, 10])

with open(os.path.dirname(os.path.realpath(__file__)) + '/semeion.data') as file:
    for line in file:
        numbers = line.split(' ')
        sample = list(map(lambda x: float(x), numbers[0:256]))
        result = list(map(lambda x: int(x), numbers[256:266]))
        sample = numpy.array([sample])
        result = numpy.array([result])
        samples = numpy.concatenate((samples, sample), axis=0)
        results = numpy.concatenate((results, result), axis=0)

first_layer = (2 * numpy.random.random((256, 256)) - 1) / 100  # the array has 256x256 dimensions
second_layer = (2 * numpy.random.random((256, 10)) - 1) / 100  # the array has 256x10 dimensions
rate = 0.4
error = 1000.0
epoch = 1
epoch_limit = 50
desired_error = 0.1

while epoch < epoch_limit and error > desired_error:
    errors = []
    for sample_index in range(samples.shape[0]):
        sample = numpy.array([samples[sample_index]])
        result = numpy.array([results[sample_index]])

        first_output = sigmoid(numpy.dot(sample, first_layer))
        second_output = sigmoid(numpy.dot(first_output, second_layer))

        second_error = result - second_output
        errors.append(numpy.max(numpy.abs(second_error)))

        second_delta = second_error * dsigmoid(second_output)
        first_error = second_delta.dot(second_layer.T)

        first_delta = first_error * dsigmoid(first_output)
        second_layer += first_output.T.dot(second_delta) * rate
        first_layer += sample.T.dot(first_delta) * rate

    error = max(errors)
    print('Epoch: %4d (of maximum %4d), max error: %.5f (of desired < %.5f)' % (epoch, epoch_limit, error, desired_error))
    epoch += 1

print('Actual testing of trained NN')

table_data = [
    ['Sample', 'Digit', 'Sample', 'Digit', 'Sample', 'Digit', 'Sample', 'Digit']
]

with open('number.jpg', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [16, 16])
        cover.save('number2.jpg', image.format)

image = img.imread('number2.jpg')
im = image
#print(image)
#print(len(image))

listfinal = []
for i in range(len(im)):
    for j,list in enumerate(im[i]):
        intensity = 0.2989*list[0] + 0.5870*list[1] + 0.1140*list[2]
        #print(str(list[0]) + "," + str(list[1]) + "," + str(list[2]) + "-" + str(intensity))
        if(intensity >= 128):
            listfinal.append(0)
        else:
            listfinal.append(1)

#print(numpy.array(listfinal))
samplesTrial = numpy.array(listfinal)

for row in range(3):
    table_data.append([''] * 8)
    for col in range(4):
        ri = random.randint(0, samples.shape[0] - 1)
        sample = samplesTrial
        table_data[row+1][col*2] = print_sample(sample)
        table_data[row+1][col*2+1] = '\n'.join([' ' * 5, ' ' * 5, '  %d' % compute_result(sample)])

table = AsciiTable(table_data)
table.inner_row_border = True

print(table.table)

