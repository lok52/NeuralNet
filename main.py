from net import *

f = open('input.txt', 'r')
net = Net([2, 4, 1])

recent_average_smoothing_factor = 4000
recent_average_error = 0

attempt_number = 0
for line in f:
    # line starts with word in
    if line[0] == 'i':
        attempt_number += 1
        print('Attempt : ', attempt_number)
        tokens = line.split()
        input_data = [float(tokens[1]), float(tokens[2])]
        print('Input : ', ' '.join(map(str, input_data)))
        net.feed_forward(input_data)
    # line starts with word out
    elif line[0] == 'o':
        tokens = line.split()
        target = [tokens[1]]
        print('Target : ', ' '.join(map(str, target)))
        print(net.get_results()[0])

        recent_average_error = (recent_average_error * recent_average_smoothing_factor + net.back_prop(target)) / (
                    recent_average_smoothing_factor + 1)

        print('Average error : ', recent_average_error)
        print()
