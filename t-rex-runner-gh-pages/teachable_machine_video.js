
// More API functions here:
// https://github.com/googlecreativelab/teachablemachine-community/tree/master/libraries/pose

// the link to your model provided by Teachable Machine export panel
const URL = "https://teachablemachine.withgoogle.com/models/lMcKKDDzw/";
let model, webcam, ctx, labelContainer, maxPredictions;

var action_keyup_event = []; 
var action_keydown_event = []; 
var prev_action_keydown_event = -1;    

async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    action_keyup_event[0] = null;
    action_keyup_event[1] = new KeyboardEvent('keyup', {keyCode: 38});
    action_keyup_event[2] = new KeyboardEvent('keyup', {keyCode: 40});
    action_keyup_event[3] = new KeyboardEvent('keyup', {keyCode: 32});
    action_keydown_event[0] = null;
    action_keydown_event[1] = new KeyboardEvent('keydown', {keyCode: 38});
    action_keydown_event[2] = new KeyboardEvent('keydown', {keyCode: 40});
    action_keydown_event[3] = new KeyboardEvent('keydown', {keyCode: 32});  

    // load the model and metadata
    // Refer to tmImage.loadFromFiles() in the API to support files from a file picker
    // Note: the pose library adds a tmPose object to your window (window.tmPose)
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Convenience function to setup a webcam
    const size = 200;
    const flip = true; // whether to flip the webcam
    webcam = new tmPose.Webcam(size, size, flip); // width, height, flip
    await webcam.setup(); // request access to the webcam
    await webcam.play();
    window.requestAnimationFrame(loop);

    // append/get elements to the DOM
    const canvas = document.getElementById("canvas");
    canvas.width = size; canvas.height = size;
    ctx = canvas.getContext("2d");
    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < maxPredictions; i++) { // and class labels
        labelContainer.appendChild(document.createElement("div"));
    }
}

async function loop(timestamp) {
    webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Prediction #1: run input through posenet
    // estimatePose can take in an image, video or canvas html element
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Prediction 2: run input through teachable machine classification model
    const prediction = await model.predict(posenetOutput);

    var max_class_index = 0;
    var max_class_prob = -1.0;
    
    for (let i = 0; i < maxPredictions; i++) {
        const classPrediction =
            prediction[i].className + ": " + prediction[i].probability.toFixed(2);
        labelContainer.childNodes[i].innerHTML = classPrediction;

        if (prediction[i].probability > max_class_prob)
        {
            max_class_prob = prediction[i].probability;
            max_class_index = i;
        }
    }

    if (prev_action_keydown_event != max_class_index && prev_action_keydown_event > 0){
        document.dispatchEvent(action_keyup_event[prev_action_keydown_event])
    }

    if (max_class_index > 0){
        document.dispatchEvent(action_keydown_event[max_class_index])
    }

    prev_action_keydown_event = max_class_index

    // finally draw the poses
    drawPose(pose);
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        // draw the keypoints and skeleton
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}
