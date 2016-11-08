import matplotlib
matplotlib.use('Agg') 
import sys
import numpy as np
import matplotlib.pyplot as plt
import lib.read_training_data as rtd

image_path    = '../result/'

def _save_one_frame(image, title):
    fig, ax = plt.subplots(sharex=True, sharey=True)
    ax.imshow(image, cmap=plt.cm.Greys_r)
    ax.axis('off')
    ax.set_title(title, fontsize=20)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                        bottom=0.02, left=0.02, right=0.98)
    plt.savefig(image_path + title)

def visualize_train_data(protobuf_path, frame_id, frame_size, title):
    _, image_list, _, _ = rtd.read_data_from_protobuf(protobuf_path)
    print >> sys.stderr, 'The len of the image list: %s' % len(image_list)
    if frame_id == -1:
        print >> sys.stderr, 'There is no specific number of the frames, plot all of the frame in the protobuf.'
        frame_id = 0
        for img_arr in image_list:
            print >> sys.stderr, 'Plot the frame %s' % frame_id
            img = img_arr.reshape(frame_size[0], frame_size[1])
            _save_one_frame(img, title + '_f' + str(frame_id))
            frame_id += 1
    else:
        print >> sys.stderr, 'Plot the frame %s' % frame_id
        img = image_list[frame_id].reshape(frame_size[0], frame_size[1])
        _save_one_frame(img, title + '_f' + str(frame_id))

def visualize_train_result():
    # Get data from stdin
    contrasts = []
    for line in sys.stdin:
        data = line.strip().split('#')
        test_output = map(float, data[0].split('\t'))
        real_output = map(float, data[1].split('\t'))
        contrasts.append(zip(test_output, real_output))
    
    # Re-organize the contrasts data
    reorg_contrasts = []
    num_contrasts = len(contrasts[0])
    for i in range(num_contrasts):
        test = []
        real = []
        for c in contrasts:
            test.append(c[i][0])
            real.append(c[i][1])
        reorg_contrasts.append([test, real])
    
    # Plot the contrasts
    x = range(len(contrasts))
    i = 0
    for con in reorg_contrasts:
        plt.figure(figsize=(50,8))
        with plt.style.context('fivethirtyeight'):
            plt.plot(x, con[0])
            plt.plot(x, con[1])
        plt.savefig(image_path + 'contrast' + str(i))
        i += 1

if __name__ == '__main__':
    # Func name:
    # - vis_data
    # - vis_result
    func_name = sys.argv[1]
    print 'function name: %s' % func_name

    if func_name == 'vis_data':
        protobuf_path = sys.argv[2]      # the path of the protobuf file
        frame_id      = int(sys.argv[3]) # if frame id is set -1, visualize all the frame in the file
        frame_size    = map(int, sys.argv[4].split(','))
        title         = sys.argv[5]

        print 'input file path:\t%s' % protobuf_path
        print 'the frame id:\t%s' % frame_id
        print 'the frame size:\t%s * %s' % tuple(frame_size)
        print 'output file name:\t%s' % title

        visualize_train_data(protobuf_path, frame_id, frame_size, title)
    elif func_name == 'vis_result':
        visualize_train_result()
