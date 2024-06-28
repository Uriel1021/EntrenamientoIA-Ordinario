import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import datos
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def cargar_y_evaluar_modelo():
    epoca = int(input("Introduce la época que deseas evaluar (número): "))
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        epoca_directorio = f"modelo_epoca_{epoca}"
        checkpoint_path = os.path.join(epoca_directorio, "modelo.ckpt")
        
        saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
        saver.restore(sess, checkpoint_path)
        
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("Placeholder:0")
        y = graph.get_tensor_by_name("Placeholder_1:0")
        output = graph.get_tensor_by_name("capa6/Relu:0")
        
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        label_test, img_test = datos.get_data("mnist_test")
        accuracy_value = sess.run(accuracy, feed_dict={x: img_test, y: label_test})
        
        predictions = np.argmax(sess.run(output, feed_dict={x: img_test}), axis=1)
        true_labels = np.argmax(label_test, axis=1)
        cm = confusion_matrix(true_labels, predictions)
        
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        
        false_positive_rate = fp / (fp + tn)
        false_negative_rate = fn / (fn + tp)
        true_positive_rate = tp / (tp + fn)
        
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], ax=axs[0])
        axs[0].set_title('Matriz de Confusión')
        axs[0].set_xlabel('Predicción')
        axs[0].set_ylabel('Valor Real')
        
        textstr = f"Exactitud: {accuracy_value:.4f}\nPrecisión: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}\n\n"
        textstr += f"Tasa de Falsos Positivos: {false_positive_rate:.4f}\n"
        textstr += f"Tasa de Falsos Negativos: {false_negative_rate:.4f}\n"
        textstr += f"Tasa de Verdaderos Positivos: {true_positive_rate:.4f}"
        
        axs[1].axis('off') 
        axs[1].text(0.1, 0.5, textstr, fontsize=12, va='center')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'accuracy': accuracy_value,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cm': cm,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positive_rate': true_positive_rate
        }

resultados = cargar_y_evaluar_modelo()
