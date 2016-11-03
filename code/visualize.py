import matplotlib
matplotlib.use('Agg') 
import sys
import numpy as np
import matplotlib.pyplot as plt
import lib.read_training_data as rtd



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
    pass

if __name__ == '__main__':
    # Func name:
    # - vis_data
    # - vis_result
    func_name = sys.argv[1]

    if func_name == 'vis_data':
        protobuf_path = sys.argv[2]      # the path of the protobuf file
        frame_id      = int(sys.argv[3]) # if frame id is set -1, visualize all the frame in the file
        frame_size    = map(int, sys.argv[4].split(','))
        title         = sys.argv[5]
        image_path    = '../result/'
        visualize_train_data(protobuf_path, frame_id, frame_size, title)
    elif func_name == 'vis_result':
        visualize_train_result()
