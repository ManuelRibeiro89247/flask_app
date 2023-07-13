from flask_restful import Resource
from flask import Flask, jsonify, request
from tensorflow import keras
import json
import os

class HandlerRes(Resource):
    def get(self):
        return {"profile": "resource"}


