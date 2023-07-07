# deep-learning-challenge
## Analysis 
The purpose of this model is to help develop a tool for nonprofit foundation Alphabet Soup to select the applicants for funding with the best chance of success in their ventures.Dataset is provided by the organization to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Steps
To achieve the purpose following steps were followed:

- After reading in the file as ```application_df ```and displaying the data, columns were displayed as follows  ```['EIN', 'NAME', 'APPLICATION_TYPE','AFFILIATION','CLASSIFICATION','USE_CASE', 'ORGANIZATION', 'STATUS','INCOME_AMT','SPECIAL_CONSIDERATIONS', 'ASK_AMT', 'IS_SUCCESSFUL']``` 
- Feature columns and target column ```IS_SUCCESSFUL``` were identified.
- Columns such as ```'EIN', 'NAME' ``` were dropped considering they do not provide information needed for further steps
- Number of unique values for each column were determined .


- For columns that have more than 10 unique values``` 'APPLICATION_TYPE','CLASSIFICATION'```, the number of data points for each unique value were determined and these were used to pick a cutoff point to bin "rare" categorical variables together in a new value, Other .

- ```pd.get_dummies(application_df)``` to encode categorical variables.

- The preprocessed data was split into a features array, X, and a target array, y. 
- These arrays were used to train_test_split function to split the data into training and testing datasets as follows ```X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,stratify = y)```.

- Scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function 
~~~scaler = StandardScaler()

X_scaler = scaler.fit(X_train)

X_train_scaled = X_scaler.transform(X_train)

X_test_scaled = X_scaler.transform(X_test)

print(len(X_train_scaled))
~~~

## Compilation,Training and evaluation
- Neural network layers were created with different combinations and summary was generated
 ~~~ nn = tf.keras.models.Sequential()

nn.add(tf.keras.layers.Dense(units = 40, activation = 'tanh', input_dim = input_features))

nn.add(tf.keras.layers.Dense(units = 60, activation = 'sigmoid'))

nn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


nn.summary()
~~~
- Compilation of the model was done 

~~~ 
nn.compile(loss="binary_crossentropy",optimizer="adam", metrics=["accuracy"])
 ~~~
  - Finally model was trained and evaluated and results were saved in ```AlphabetSoupCharity.h5.```
  ~~~
  fit_model = nn.fit(X_train_scaled, y_train, epochs= 50, callbacks=[checkpoint])
  model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
  ~~~

  ## Results