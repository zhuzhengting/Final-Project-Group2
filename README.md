# Final-Project-Group2
prediction on Titanic survival

the description of data
Variable	Definition	Key
Survival	Survival	0=No; 1=Yes
Pclass	Ticket Class	1=1st (Upper); 2=2nd (Middle); 3=3rd (Lower)
Sex	Sex	
Age	Age in years	
Sibsp	# of siblings / spouses aboard the Titanic	"Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)"
Parch	# of parents / children aboard the Titanic	"Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.
"
Ticket	Ticket Number	
Fare	Passenger fare	
Cabin	Cabin Number	
Embarked	Port of Embarkation	C= Cherbourg, Q= Queenstown, S=Southampton

The order of codes:

1, codes to import data should be run first.

2, then codes to preprocess then replace missing characters as NaN, replace categorical mushroom_data with the most frequent value in that column, drop Cabin, ticket and name of passenger and finally encode features and normalize the data.

3, codes to set up decision tree using gini and entropy and create metrics report, confusion metrics and plot decision tree.

4, codes to set up support vector machine and create metrics report, confusion metrics and ROC curve

5, codes to set up naive bayes and create metrics report and confusion metrics
