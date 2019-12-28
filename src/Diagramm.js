// in Anlehnung vom Beispielcode von Tensorflow
import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// Lineare Regression
class Diagramm extends Component {

    constructor(props) {
        super(props);


    }
    componentDidMount() {
        this.run();
    }
    /**
 * Gibt alle Daten aus, die Meilen und Gallone enthalten
 * der Rest wird gelöscht
 */
    async getData() {
        const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
        const carsData = await carsDataReq.json();
        const cleaned = carsData.map(car => ({
            mpg: car.Miles_per_Gallon,
            horsepower: car.Horsepower,
        }))
            .filter(car => (car.mpg != null && car.horsepower != null));
        console.log(cleaned);
        return cleaned;
    }
    // startet das Programm
    async run() {
        // Laden Sie die ursprünglichen Eingabedaten, mit denen wir trainieren, und zeichnen Sie sie auf.
        const data = await this.getData();
        const values = data.map(d => ({
            x: d.horsepower,
            y: d.mpg,
        }));

        tfvis.render.scatterplot(
            { name: 'Horsepower v MPG' },
            { values },
            {
                xLabel: 'Horsepower',
                yLabel: 'MPG',
                height: 300
            }
        );

        // ---------------------More code will be added below
        // Create the model
        const model = this.createModel();
        tfvis.show.modelSummary({ name: 'Model Summary' }, model);

        // Convert the data to a form we can use for training.
        const tensorData = this.convertToTensor(data);
        const { inputs, labels } = tensorData;
        // Train the model  
        await this.trainModel(model, inputs, labels);
        console.log('Done Training');

        // Machen Sie anhand des Modells einige Vorhersagen und vergleichen Sie sie mit den Originaldaten
        this.testModel(model, data, tensorData);
    }
    createModel() {
        // ein Model instanziieren=> sequential eingang direkt zu ausgang
        // tfvis.visor().el=document.getElementById('boardID');
        document.getElementById('boardID').appendChild(tfvis.visor().el);
        const model = tf.sequential();


        // Ebenen hinzufügen (dense=>Matrix multipliziert, Ergebnis Zahl(Bias))
        // inputShape: Schicht (hier erste), units: Gewichtungsmatrix, useBias: Ergebnis
        model.add(tf.layers.dense({ inputShape: [1], units: 50, useBias: true }));


        // Add an output layer
        model.add(tf.layers.dense({ units: 1, useBias: true }));



        return model;
    }
    /**
    * 
    *  Konvertieren Sie die Eingabedaten in Tensoren, die wir für die Maschine verwenden können
    *  Lernen. Wir werden auch die wichtigen Best Practices von _shuffling_ ausführen.
    *  die Daten und _normalizing_ die Daten
    *  MPG auf der y-Achse.
    */
    convertToTensor(data) {
        // Wenn Sie diese Berechnungen ordentlich einpacken, 
        // werden alle dazwischenliegenden Tensoren entfernt.

        return tf.tidy(() => {
            //------------ Step 1. Shuffle the data    
            tf.util.shuffle(data);

            // ------------Step 2. Convert data to Tensor
            // Hier erstellen wir zwei Arrays, eines für unsere 
            // Eingabebeispiele (die PS-Einträge) und eines für die tatsächlichen 
            // Ausgabewerte (die beim maschinellen Lernen als Labels bezeichnet werden).

            // Wir konvertieren dann alle Array-Daten in einen 2D-Tensor. 
            // Der Tensor hat eine Form von [num_examples, num_features_per_example]. 
            // Hier haben wir inputs.lengthBeispiele und jedes Beispiel hat eine 1Eingabefunktion (die Pferdestärke).
            const inputs = data.map(d => d.horsepower)
            const labels = data.map(d => d.mpg);

            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            //--------------Step 3. Normalize the data to the range 0 - 1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();

            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
            // die Normalisierung wird zurückgegeben
            return {
                inputs: normalizedInputs,
                labels: normalizedLabels,
                // Geben Sie die Min / Max-Grenzen zurück, damit wir sie später verwenden können.
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            }
        });
    }
    async trainModel(model, inputs, labels) {
        // Prepare the model for training.  
        model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ['mse'],
        });
        // optimizer: Dies ist der Algorithmus, der die Aktualisierungen des Modells anhand von Beispielen 
        // steuert. In TensorFlow.js sind viele Optimierer verfügbar. Hier haben wir den Adam Optimizer ausgewählt, 
        // da er in der Praxis sehr effektiv ist und keine Konfiguration erfordert.
        // loss: Dies ist eine Funktion, die dem Modell mitteilt, wie gut es beim Lernen der 
        // einzelnen Stapel (Datenteilmengen), die angezeigt werden, vorgeht. Hier meanSquaredErrorvergleichen wir 
        // die vom Modell gemachten Vorhersagen mit den wahren Werten.


        // batchSize bezieht sich auf die Größe der Datenteilmengen, die das Modell bei jeder Trainingsiteration sieht. 
        // Übliche Losgrößen liegen in der Regel im Bereich von 32 bis 512. Es gibt nicht wirklich eine ideale Losgröße 
        // für alle Probleme und es würde den Rahmen dieses Lernprogramms sprengen, die mathematischen Beweggründe für 
        // verschiedene Losgrößen zu beschreiben.
        // epochs bezieht sich auf die Häufigkeit, mit der das Modell auf den gesamten von Ihnen bereitgestellten Datensatz 
        // zugreift. Hier nehmen wir 50 Iterationen durch den Datensatz.
        const batchSize = 32;
        const epochs = 50;


        // model.fitist die Funktion, die wir aufrufen, um die Trainingsschleife zu starten. Da es sich um eine asynchrone 
        // Funktion handelt, geben wir das von ihr gegebene Versprechen zurück, damit der Anrufer feststellen kann, wann das Training abgeschlossen ist.
        // Um den Trainingsfortschritt zu überwachen, leiten wir einige Rückrufe an weiter model.fit. Wir tfvis.show.fitCallbacksgenerieren damit 
        // Funktionen, die Diagramme für die zuvor angegebenen Metriken 'loss' und 'mse' zeichnen.
        return await model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance' },
                ['loss', 'mse'],
                { height: 200, callbacks: ['onEpochEnd'] }
            )
        });
    }
    testModel(model, inputData, normalizationData) {
        const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

        // Generate predictions for a uniform range of numbers between 0 and 1;
        // We un-normalize the data by doing the inverse of the min-max scaling 
        // that we did earlier.
        const [xs, preds] = tf.tidy(() => {



            // Wir generieren 100 neue "Beispiele", die dem Modell hinzugefügt werden sollen. Model.predict ist, wie 
            // wir diese Beispiele in das Modell einspeisen. Beachten Sie, dass sie eine ähnliche Form haben müssen 
            // ( [num_examples, num_features_per_example]) wie beim Training.
            const xs = tf.linspace(0, 1, 100);//erzeugen die 100 werte
            const preds = model.predict(xs.reshape([100, 1]));//erzeugen 100 werte Achtung: Immer beide 100 von linespace und prdict wechseln

            // Um die Daten wieder in den ursprünglichen Bereich (anstelle von 0-1) zu bringen, verwenden wir die Werte, 
            // die wir beim Normalisieren berechnet haben, kehren aber nur die Operationen um.
            const unNormXs = xs
                .mul(inputMax.sub(inputMin))
                .add(inputMin);

            const unNormPreds = preds
                .mul(labelMax.sub(labelMin))
                .add(labelMin);

            // Un-normalize the data
            // .dataSync()Mit dieser Methode können wir einen typedarray der in einem Tensor gespeicherten Werte 
            // ermitteln. Dadurch können wir diese Werte in regulärem JavaScript verarbeiten. Dies ist eine synchrone 
            // Version der .data()Methode, die im Allgemeinen bevorzugt wird.
            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });


        const predictedPoints = Array.from(xs).map((val, i) => {
            return { x: val, y: preds[i] }
        });

        const originalPoints = inputData.map(d => ({
            x: d.horsepower, y: d.mpg,
        }));


        let tmp = tfvis.render.scatterplot(
            // { name: 'Model Predictions vs Original Data' },
            document.getElementById('testID'),//hier wurde ein neues Div element erstellt um diese Ausgabe dort zu plazieren
            { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
            {
                xLabel: 'Horsepower',
                yLabel: 'MPG',
                height: 300
            }
        );
        console.log(tmp);
    }

    //----------------------------------------------------------------------------------
    render() {



        return (
            <div>
                <div className="board" id="boardID">
                    Hello

                </div>
                <div id="testID">

                </div>
            </div>
        );
    }
}
export default Diagramm;