// more documentation available at
// https://github.com/tensorflow/tfjs-models/tree/master/speech-commands

// the link to your model provided by Teachable Machine export panel
const URL = "https://teachablemachine.withgoogle.com/models/BMNKsfgL8/";
//const URL = "/Users/samuelkim/Desktop/t-rex-runner-gh-pages/audio_model/";

var action_keyup_event = []; 
var action_keydown_event = []; 
var prev_action_keydown_event = -1;

async function createModel() {
    const checkpointURL = URL + "model.json"; // model topology
    const metadataURL = URL + "metadata.json"; // model metadata

    const recognizer = speechCommands.create(
        "BROWSER_FFT", // fourier transform type, not useful to change
        undefined, // speech commands vocabulary feature, not useful for your models
        checkpointURL,
        metadataURL);

    // check that model and metadata are loaded via HTTPS requests.
    await recognizer.ensureModelLoaded();

    return recognizer;
}

async function init() {
    const recognizer = await createModel();
    const classLabels = recognizer.wordLabels(); // get class labels

    action_keyup_event[0] = null;
    action_keyup_event[1] = new KeyboardEvent('keyup', {keyCode: 38});
    action_keyup_event[2] = new KeyboardEvent('keyup', {keyCode: 40});
    action_keyup_event[3] = new KeyboardEvent('keyup', {keyCode: 32});
    action_keydown_event[0] = null;
    action_keydown_event[1] = new KeyboardEvent('keydown', {keyCode: 38});
    action_keydown_event[2] = new KeyboardEvent('keydown', {keyCode: 40});
    action_keydown_event[3] = new KeyboardEvent('keydown', {keyCode: 32});            

    const labelContainer = document.getElementById("label-container");
    for (let i = 0; i < classLabels.length; i++) {
        labelContainer.appendChild(document.createElement("div"));
    }
    labelContainer.appendChild(document.createElement("div"));

    // listen() takes two arguments:
    // 1. A callback function that is invoked anytime a word is recognized.
    // 2. A configuration object with adjustable fields       
    recognizer.listen(result => {
        var max_class_index = 0;
        var max_class_prob = -1.0;
        const scores = result.scores; // probability of prediction for each class
        // render the probability scores per class
        for (let i = 0; i < classLabels.length; i++) {
            const classPrediction = classLabels[i] + ": " + result.scores[i].toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;

            if ( result.scores[i].toFixed(2) > max_class_prob )
            {
                max_class_prob = result.scores[i];
                max_class_index = i;     
            }
        }
        
        labelContainer.childNodes[classLabels.length].innerHTML = classLabels[max_class_index];

        if (prev_action_keydown_event != max_class_index && prev_action_keydown_event > 0){
            document.dispatchEvent(action_keyup_event[prev_action_keydown_event])
        }

        if (max_class_index > 0){
            document.dispatchEvent(action_keydown_event[max_class_index])
        }

        prev_action_keydown_event = max_class_index

        
    }, {
        includeSpectrogram: true, // in case listen should return result.spectrogram
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.50 // probably want between 0.5 and 0.75. More info in README
    });

    // Stop the recognition in 5 seconds.
    // setTimeout(() => recognizer.stopListening(), 5000);
}