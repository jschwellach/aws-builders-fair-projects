from inference import inference

if __name__ == "__main__":
    inference = inference.Inference(src=0, resolution=(
        1280, 720), 
        detection_rate=1, mode='TENSORFLOW', context='gpu',
        model='model/frozen_inference_graph.pb', labels=['Fish', 'Tin Can', 'Plastic Bag'])
    inference.run()
