###XGBOOST:

Fit time 67.29 s

Model params:

        max_depth=10
        learning_rate=0.1
        n_estimators=50
        objective='reg:squarederror'
        reg_alpha=10
        reg_lambda=10

Train:

        Grid best score: -462225.60621526535, with params: {'learning_rate': 0.1, 'max_depth': 20, 'objective': 'reg:squarederror', 'reg_alpha': 10}
        Model train score: [-432688.43059788 -480341.55148575 -451711.80581077 -454236.10998566 -508930.10225999]
        Train mse: 1374734331966.571
        Test mse: 3020492338902.609

###Torch NN:

Fit time: 39.948505878448486 s

Model params:

    batch_size=45
    hidden_dim=20
    hidden_num=3
    alpha=0.01 
    epochs=100

Train:

    Train mse: 4620279705910.527
    Test mse: 7927956529745.623

###Testing:
`POST /api/v1/predict`

Body:
```
{
	"model": "fnn",
	"features": {
		"total_square_meters": 50, 
		"living_square_meters": 25, 
		"kitchen_square_meters": 10,
		"rooms_count": 1, 
		"floor": 10,
        "construction_year": 2001, 
        "floors_count": 10, 
        "inspected": 1,
        "street_name": "улица", 
        "city_name": "Калуш", 
        "wall_type": "блочно-кирпичный", 
        "heating": "централизованное", 
        "seller": "от собственника", 
        "water": "централизованное (водопровод)", 
        "building_condition": "нормальное"
	}
}
```
Response:
```
{
  "inference_time": 0.0002770423889160156,
  "predicted_price": 1831373.0
}
```

`POST /api/v1/predict`

Body:
```
{
	"model": "xgboost",
	"features": {
		"total_square_meters": 100, 
		"living_square_meters": 15, 
		"kitchen_square_meters": 20,
		"rooms_count": 3, 
		"floor": 4,
        "construction_year": 2009, 
        "floors_count": 14, 
        "inspected": 1,
        "street_name": "проспект", 
        "city_name": "Киев", 
        "wall_type": "блочно-кирпичный", 
        "heating": "централизованное", 
        "seller": "от собственника", 
        "water": "централизованное (водопровод)", 
        "building_condition": "отличное"
	}
}
```
Response:
```
{
  "inference_time": 0.009671926498413086,
  "predicted_price": 506259.5625
}
```

`GET /api/v1/statistics`

Response:
```
{
  "mean_price": "1250647.79", 
  "mean_total_square_meters": "5477.30", 
  "std_price": "2293839.02", 
  "std_total_square_meters": "570842.22", 
  "total_apartments_number": 15495
}
```
