<h1>🖥️ MNIST Classification Using Multiple Machine Learning Models</h1>

<p>
This project implements and compares several machine-learning models to classify handwritten digits from the 
<strong>MNIST dataset</strong>. It includes training, evaluation, ROC curve analysis, and hyperparameter tuning 
using GridSearchCV (for SVM).
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>The project loads MNIST train and test CSV files, trains multiple ML classifiers, evaluates their performance, and plots ROC curves for each model.</p>

<h3>Models Implemented</h3>
<ul>
    <li>K-Nearest Neighbors (KNN)</li>
    <li>Logistic Regression</li>
    <li>Naive Bayes (GaussianNB)</li>
    <li>Decision Tree</li>
    <li>Random Forest</li>
    <li>AdaBoost</li>
    <li>Gradient Boosting</li>
    <li>XGBoost</li>
    <li>Support Vector Machine (SVM)</li>
    <li>SVM (with Grid Search Optimization)</li>
</ul>

<p>Each model outputs:</p>
<ul>
    <li>Test Accuracy</li>
    <li>Confusion Matrix</li>
    <li>Classification Report</li>
    <li>ROC Curve (macro-averaged across all 10 classes)</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
MNIST-ML-Project
│── mnist_train.csv
│── mnist_test.csv
│── main.py
│── README.md
└── requirements.txt
</pre>

<hr>

<h2>⚙️ How the Code Works</h2>

<h3>1. MNIST Class Initialization</h3>
<p>Loads training and testing data and splits into:</p>
<ul>
    <li>X_train, y_train</li>
    <li>X_test, y_test</li>
</ul>

<h3>2. Model Training Methods</h3>
<p>Each algorithm has its own function such as <code>knn()</code>, <code>lg()</code>, <code>nb()</code>, etc.</p>

<h3>3. Grid Search for SVM</h3>
<p>The method <code>svm_grid()</code> performs:</p>
<ul>
    <li>Parameter search using GridSearchCV</li>
    <li>Selects best parameters</li>
    <li>Trains optimized SVM model</li>
</ul>

<h3>4. ROC Curve Calculation</h3>
<p>Uses <code>label_binarize</code> and macro-averages ROC across all classes.</p>

<h3>5. Final ROC Plot</h3>
<p>A combined ROC comparison of all ML models.</p>

<hr>

<h2>▶️ How to Run the Project</h2>

<h3>1. Clone the Repository</h3>
<pre>
git clone https://github.com/yourusername/MNIST-ML-Project.git
cd MNIST-ML-Project
</pre>

<h3>2. Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>3. Run the Script</h3>
<pre>
python main.py
</pre>

<hr>

<h2>📊 Results</h2>
<p>The script prints:</p>
<ul>
    <li>Accuracy of each model</li>
    <li>Confusion matrices</li>
    <li>Classification reports</li>
    <li>ROC curves comparing all models</li>
</ul>

<p>SVM with GridSearch usually performs best.</p>

<hr>

<h2>🧪 Requirements</h2>

<pre>
numpy
pandas
matplotlib
scikit-learn
xgboost
</pre>

<hr>

<h2>📝 Notes</h2>
<ul>
    <li>Uses CSV-based MNIST, not image files.</li>
    <li>Ensure dataset paths are correct inside <code>main.py</code>.</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>
<p>Pull requests are welcome for improvements such as hyperparameter tuning, visualization enhancements, or adding deep learning models.</p>

<hr>

<h2>📄 License</h2>
<p>This project is licensed under the MIT License.</p>
