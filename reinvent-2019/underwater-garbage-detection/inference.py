from inference import inference

if __name__ == "__main__":
    '''Calling the inference application
    make sure to select the right src for your webcam'''
    inference = inference.Inference(src=0, resolution=(
        1280, 720), 
        detection_rate=1, mode='TENSORFLOW', context='gpu',
        model='model/frozen_inference_graph.pb', labels=['Fish', 'Tin Can', 'Plastic Bag'])
    inference.run()
