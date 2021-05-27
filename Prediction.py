import torch
from main import FeedForward, download_datasets

class_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        prediction = model(input)

        #         tensor object is stroed in predictions (1, 10) size
        predicted_index = prediction[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == '__main__':
    # load the model
    feed_forward_net = FeedForward()
    state_dict = torch.load('/PATH OF THE MODEL')

    feed_forward_net.load_state_dict(state_dict)

    #     download validation dataset
    _, validation = download_datasets()
    #     get a sample for inference
    input, target = validation[0][0], validation[0][1]

    #     make the inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)
    print("Predicted = {} expected = {}".format(predicted, expected))
