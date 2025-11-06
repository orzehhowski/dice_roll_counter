## dice roll counter

Sometimes I play board games. Board games are usually combination of skill and luck, what leads to some problems in determining who's better. But if there was a system measuring luck, it would clarify a lot - and that's my goal. 

### How will it work

System will consist of several elements:

1. Dice roll arena with raspberry pi slot - printed on 3D printer
2. Raspberry pi with camera, some screens and buttons
3. ML model counting pips on dice face
4. Python API hosted somewhere, accepting image and returning dice roll results
5. Python client app, sending data to API or using local model if API is unaccessible and counting game's overalls
6. [Optional] Python service that counts long-term statistics for players

So, I think that project is pretty straight-forward - roll the dice, take a photo, send it to API, gather results, add it to overalls and save in statistics service. 
