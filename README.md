## Setting Up Repo for Test Generation
1. clone the repository.
2. install necessary packages(refer to REQUIREMENTS.txt)
3. install [Appium](http://appium.io/) Desktop
4. add app's configuration in [App_Config.py] 
    -- you can find relevant info in [news_app_info.csv], [shopping_app_info.csv], and [game_app_info.csv]
5. start Android Emulator -- make sure the AUT is installed on the Emulator, e.g., abcnews.
6. start Appium Server on Appium desktop.
7. change file paths in [global_config.py].
8. change the number of generated tests in [test_generator_auto.py](Modify self.MAX_TEST_NUM=2 in the [class TestGenerator] if you want to generate two tests. )
9. run main method in [test_generator_auto.py]


[`Dynamic Test Generation`]: Dynamically generating usage-based tests for new apps
Notice:
 the definitions of canonical screens and canonical widgets are under [`IR`] folder
 the IR Models, generated tests, and intermediate results (e.g., screenshots, cropped widgets, reverse engineered UI layout hierarchy) are under [`Final-Artifacts\output`] folder.
 the processed video frames, screenshots, cropped widgets, keyboard classifier's results of all the usages are under [`usage_data`] folder (refer to Avgust for evaluation).


At this point you should be able to see an output similar to the output below on your command line interface:

>>The screen classifier top5 guesses for the screen:
['category', 'home', 'items', 'menu', 'popup']
Choose the closest screen tag from the top5 guesses: [`You should type the closest screen tag on your cli`]
`home`
---------
>>id:0 floating_search_view - matched with: to_search
id:1 rl_search_box - matched with: to_search
id:2 navigation_home - matched with: home
id:3 navigation_feed - matched with: menu
Choose the id of the widget you want to interact with:[`You should type the id on your cli`]: `0`
Please enter the ground truth IR for the widget you chose:to_search
executing event: resource-id floating_search_view click
