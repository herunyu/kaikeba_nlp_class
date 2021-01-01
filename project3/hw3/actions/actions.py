# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import requests

from rasa_sdk import Action, Tracker
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.executor import CollectingDispatcher


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []

class ActionAskWeather(Action):
   def name(self) -> Text:
      return "action_ask_weather"

   def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        city = tracker.get_slot('city')
        res = requests.get("http://www.tianqiapi.com/free/day?version=v9&appid=46294446&appsecret=KeyqNE25&city=" + city)
        if res.ok:
            weather_detail = res.json()
        else:
            dispatcher.utter_message(text='系统错误')
            return []
        response = "{}今天天气{}， 当前气温{}，日间气温可达{}，晚间气温可达{}，吹{}，风力{}，空气指数{}".format(weather_detail['city'], weather_detail['wea'], weather_detail['tem'], weather_detail['tem_day'], weather_detail['tem_night'], weather_detail['win'], weather_detail['win_meter'], weather_detail['air'])
        # q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
        # result = db.query(q)
        dispatcher.utter_message(text=response)

        return [AllSlotsReset()]

# class ActionAskWeather(Action):
#    def name(self) -> Text:
#       return "action_ask_weather"

#    def run(self,
#            dispatcher: CollectingDispatcher,
#            tracker: Tracker,
#            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#       city = tracker.get_slot('city')
#       q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
#       result = db.query(q)

#       return [SlotSet("matches", result if result is not None else [])]