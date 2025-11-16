


import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
from io import BytesIO

food_metadata = {
    "sausage":             {"kcal_per_100g": 300, "protein_g": 13.0, "fat_g": 27.0, "carb_g": 1.0,  "vitamins": "B12, B6",    "type": "countable", "avg_weight_g": 50},
    "fried_egg":           {"kcal_per_100g": 196, "protein_g": 13.0, "fat_g": 15.0, "carb_g": 1.0,  "vitamins": "A, D, B12",  "type": "countable", "avg_weight_g": 55},
    "nuggets":             {"kcal_per_100g": 280, "protein_g": 15.0, "fat_g": 18.0, "carb_g": 15.0, "vitamins": "B3, B6",     "type": "countable", "avg_weight_g": 18},
    "Chinese_sausage":     {"kcal_per_100g": 400, "protein_g": 24.0, "fat_g": 45.0, "carb_g": 8.0,  "vitamins": "B1, B12",    "type": "countable", "avg_weight_g": 40},
    "pork_balls":          {"kcal_per_100g": 200, "protein_g": 15.0, "fat_g": 20.0, "carb_g": 5.0,  "vitamins": "B12",        "type": "countable", "avg_weight_g": 15},
    "Crab_stick":          {"kcal_per_100g": 95,  "protein_g": 8.0,  "fat_g": 0.5,  "carb_g": 15.0, "vitamins": "B12, Selenium","type": "countable", "avg_weight_g": 16},
    "Five_Spice_Egg":      {"kcal_per_100g": 150, "protein_g": 13.0, "fat_g": 10.0, "carb_g": 1.5,  "vitamins": "A, D",       "type": "countable", "avg_weight_g": 60},
    "omelette":            {"kcal_per_100g": 155, "protein_g": 11.0, "fat_g": 12.0, "carb_g": 1.0,  "vitamins": "A, D, B12",  "type": "countable", "avg_weight_g": 90},
    "sai_ua":              {"kcal_per_100g": 350, "protein_g": 15.0, "fat_g": 30.0, "carb_g": 5.0,  "vitamins": "B1, B3",     "type": "countable", "avg_weight_g": 150},
    "rice":                {"kcal_per_100g": 130, "protein_g": 2.7,  "fat_g": 0.3,  "carb_g": 28.0, "vitamins": "B1, B3",     "type": "bulk",      "avg_weight_g": 200},
    "fried_rice":          {"kcal_per_100g": 180, "protein_g": 5.0,  "fat_g": 7.0,  "carb_g": 25.0, "vitamins": "B1, B3",     "type": "bulk",      "avg_weight_g": 250},
    "ham":                 {"kcal_per_100g": 145, "protein_g": 18.0, "fat_g": 6.0,  "carb_g": 1.5,  "vitamins": "B1, B12",    "type": "countable", "avg_weight_g": 30},
    "fried_chicken_legs":  {"kcal_per_100g": 260, "protein_g": 25.0, "fat_g": 17.0, "carb_g": 0.0,  "vitamins": "B6, B3",     "type": "countable", "avg_weight_g": 120},
    "fried_chicken":       {"kcal_per_100g": 240, "protein_g": 27.0, "fat_g": 14.0, "carb_g": 0.0,  "vitamins": "B6, B3",     "type": "countable", "avg_weight_g": 100},
    "fried_chicken_wings": {"kcal_per_100g": 290, "protein_g": 30.0, "fat_g": 20.0, "carb_g": 0.0,  "vitamins": "B6, B3",     "type": "countable", "avg_weight_g": 80},
    "mixed_vegetable":     {"kcal_per_100g": 35,  "protein_g": 2.0,  "fat_g": 0.2,  "carb_g": 7.0,  "vitamins": "C, A, K",    "type": "bulk",      "avg_weight_g": 150},
    "pork_stewed":         {"kcal_per_100g": 250, "protein_g": 20.0, "fat_g": 18.0, "carb_g": 2.0,  "vitamins": "B1, B12",    "type": "bulk",      "avg_weight_g": 100},
    "pork_stirfried":      {"kcal_per_100g": 280, "protein_g": 22.0, "fat_g": 20.0, "carb_g": 3.0,  "vitamins": "B3, B6",     "type": "bulk",      "avg_weight_g": 100},
    "pork_curry":          {"kcal_per_100g": 230, "protein_g": 15.0, "fat_g": 15.0, "carb_g": 8.0,  "vitamins": "A, C",       "type": "bulk",      "avg_weight_g": 100},
    "crispy_pork":         {"kcal_per_100g": 550, "protein_g": 12.0, "fat_g": 55.0, "carb_g": 0.0,  "vitamins": "B12",        "type": "countable", "avg_weight_g": 50},
    "sweet_pork":          {"kcal_per_100g": 300, "protein_g": 18.0, "fat_g": 20.0, "carb_g": 10.0, "vitamins": "B1",         "type": "countable", "avg_weight_g": 50},
    "pork_scratchings":    {"kcal_per_100g": 600, "protein_g": 30.0, "fat_g": 55.0, "carb_g": 0.0,  "vitamins": "B12",        "type": "countable", "avg_weight_g": 15},
    "mooyor":              {"kcal_per_100g": 300, "protein_g": 13.0, "fat_g": 27.0, "carb_g": 1.0,  "vitamins": "B12, B6",    "type": "countable", "avg_weight_g": 50},
    "minced_pork":         {"kcal_per_100g": 290, "protein_g": 25.0, "fat_g": 20.0, "carb_g": 0.0,  "vitamins": "B3, B6, B12","type": "bulk",      "avg_weight_g": 80},
    "boiled_pork":         {"kcal_per_100g": 200, "protein_g": 27.0, "fat_g": 10.0, "carb_g": 0.0,  "vitamins": "B6, B12",    "type": "countable", "avg_weight_g": 100},
    "boiled_meat":         {"kcal_per_100g": 180, "protein_g": 26.0, "fat_g": 8.0,  "carb_g": 0.0,  "vitamins": "B6, B12",    "type": "countable", "avg_weight_g": 100},
    "shrimp_stirfried":    {"kcal_per_100g": 200, "protein_g": 20.0, "fat_g": 10.0, "carb_g": 5.0,  "vitamins": "B12, D",     "type": "bulk",      "avg_weight_g": 100},
    "shrimp_fried":        {"kcal_per_100g": 260, "protein_g": 18.0, "fat_g": 15.0, "carb_g": 12.0, "vitamins": "B12, D",     "type": "countable", "avg_weight_g": 100},
    "fried_crab":          {"kcal_per_100g": 250, "protein_g": 18.0, "fat_g": 16.0, "carb_g": 8.0,  "vitamins": "B12, Zinc",  "type": "countable", "avg_weight_g": 100},
    "crab_meat":           {"kcal_per_100g": 80,  "protein_g": 18.0, "fat_g": 1.0,  "carb_g": 0.0,  "vitamins": "B12, Zinc",  "type": "bulk",      "avg_weight_g": 50},
    "fish_steamed":        {"kcal_per_100g": 120, "protein_g": 22.0, "fat_g": 3.0,  "carb_g": 0.0,  "vitamins": "D, B12",     "type": "countable", "avg_weight_g": 100},
    "grilled_fish":        {"kcal_per_100g": 160, "protein_g": 25.0, "fat_g": 6.0,  "carb_g": 0.0,  "vitamins": "D, B12",     "type": "countable", "avg_weight_g": 100},
    "fish_maw":            {"kcal_per_100g": 25,  "protein_g": 5.0,  "fat_g": 0.2,  "carb_g": 1.0,  "vitamins": "Iron",       "type": "bulk",      "avg_weight_g": 50},
    "perna_viridis":       {"kcal_per_100g": 172, "protein_g": 24.0, "fat_g": 4.5,  "carb_g": 8.0,  "vitamins": "B12, Iron",  "type": "countable", "avg_weight_g": 100},
    "opyster":             {"kcal_per_100g": 68,  "protein_g": 7.0,  "fat_g": 2.5,  "carb_g": 4.0,  "vitamins": "B12, Zinc",  "type": "countable", "avg_weight_g": 50},
    "blood_clam":          {"kcal_per_100g": 110, "protein_g": 14.0, "fat_g": 1.5,  "carb_g": 9.0,  "vitamins": "Iron, B12",  "type": "countable", "avg_weight_g": 50},
    "mantis_shrimp":       {"kcal_per_100g": 90,  "protein_g": 20.0, "fat_g": 1.0,  "carb_g": 0.0,  "vitamins": "B12",        "type": "countable", "avg_weight_g": 50},
    "squid":               {"kcal_per_100g": 92,  "protein_g": 16.0, "fat_g": 1.5,  "carb_g": 3.0,  "vitamins": "B12, C",     "type": "countable", "avg_weight_g": 100},
    "squid_ring":          {"kcal_per_100g": 120, "protein_g": 15.0, "fat_g": 2.0,  "carb_g": 10.0, "vitamins": "B12, C",     "type": "countable", "avg_weight_g": 50},
    "squid_steamed":       {"kcal_per_100g": 90,  "protein_g": 16.0, "fat_g": 1.5,  "carb_g": 3.0,  "vitamins": "B12, C",     "type": "countable", "avg_weight_g": 100},
    "fried_squid":         {"kcal_per_100g": 150, "protein_g": 14.0, "fat_g": 7.0,  "carb_g": 8.0,  "vitamins": "B12, C",     "type": "countable", "avg_weight_g": 100},
    "fried_wonton":        {"kcal_per_100g": 330, "protein_g": 10.0, "fat_g": 20.0, "carb_g": 25.0, "vitamins": "B1, B3",     "type": "countable", "avg_weight_g": 30},
    "Shumai":              {"kcal_per_100g": 150, "protein_g": 10.0, "fat_g": 8.0,  "carb_g": 10.0, "vitamins": "B12",        "type": "countable", "avg_weight_g": 30},
    "spring_roll":         {"kcal_per_100g": 200, "protein_g": 6.0,  "fat_g": 10.0, "carb_g": 20.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 50},
    "fresh_spring_roll":   {"kcal_per_100g": 80,  "protein_g": 5.0,  "fat_g": 2.0,  "carb_g": 10.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 50},
    "fried_tofu":          {"kcal_per_100g": 265, "protein_g": 10.0, "fat_g": 20.0, "carb_g": 10.0, "vitamins": "Calcium",    "type": "countable", "avg_weight_g": 100},
    "egg_tofu":            {"kcal_per_100g": 80,  "protein_g": 8.0,  "fat_g": 5.0,  "carb_g": 1.0,  "vitamins": "Calcium",    "type": "countable", "avg_weight_g": 50},
    "half_boiled_egg":     {"kcal_per_100g": 155, "protein_g": 13.0, "fat_g": 11.0, "carb_g": 1.1,  "vitamins": "A, D, B12",  "type": "countable", "avg_weight_g": 60},
    "egg_scrambled":       {"kcal_per_100g": 150, "protein_g": 10.0, "fat_g": 11.0, "carb_g": 1.5,  "vitamins": "A, D, B12",  "type": "bulk",      "avg_weight_g": 100},
    "steamed_egg":         {"kcal_per_100g": 120, "protein_g": 11.0, "fat_g": 8.0,  "carb_g": 1.0,  "vitamins": "A, D, B12",  "type": "countable", "avg_weight_g": 100},
    "chicken_meat_steamed":{"kcal_per_100g": 150, "protein_g": 25.0, "fat_g": 5.0,  "carb_g": 0.0,  "vitamins": "B6, B3",     "type": "countable", "avg_weight_g": 100},
    "chicken_stirfried":   {"kcal_per_100g": 200, "protein_g": 20.0, "fat_g": 12.0, "carb_g": 3.0,  "vitamins": "B6, B3",     "type": "bulk",      "avg_weight_g": 100},
    "chicken_thighs":      {"kcal_per_100g": 210, "protein_g": 26.0, "fat_g": 11.0, "carb_g": 0.0,  "vitamins": "B6, B12",    "type": "countable", "avg_weight_g": 100},
    "chicken_legs":        {"kcal_per_100g": 200, "protein_g": 28.0, "fat_g": 9.0,  "carb_g": 0.0,  "vitamins": "B6, B12",    "type": "countable", "avg_weight_g": 100},
    "chicken_wings":       {"kcal_per_100g": 290, "protein_g": 30.0, "fat_g": 20.0, "carb_g": 0.0,  "vitamins": "B6, B3",     "type": "countable", "avg_weight_g": 80},
    "pad_see_eiw_noodle":  {"kcal_per_100g": 190, "protein_g": 7.0,  "fat_g": 8.0,  "carb_g": 22.0, "vitamins": "B1, B3",     "type": "bulk",      "avg_weight_g": 200},
    "pad_thai":            {"kcal_per_100g": 300, "protein_g": 10.0, "fat_g": 12.0, "carb_g": 38.0, "vitamins": "C, B1",      "type": "bulk",      "avg_weight_g": 200},
    "yellow_noodle":       {"kcal_per_100g": 370, "protein_g": 14.0, "fat_g": 5.0,  "carb_g": 75.0, "vitamins": "B1, B3",     "type": "bulk",      "avg_weight_g": 100},
    "big_flat_noodle":     {"kcal_per_100g": 350, "protein_g": 8.0,  "fat_g": 1.0,  "carb_g": 78.0, "vitamins": "B1",         "type": "bulk",      "avg_weight_g": 100},
    "glass_noodle":        {"kcal_per_100g": 340, "protein_g": 0.2,  "fat_g": 0.1,  "carb_g": 86.0, "vitamins": "Iron",       "type": "bulk",      "avg_weight_g": 100},
    "noodles":             {"kcal_per_100g": 350, "protein_g": 8.0,  "fat_g": 1.0,  "carb_g": 78.0, "vitamins": "B1",         "type": "bulk",      "avg_weight_g": 100},
    "khao_kluk_kapi":      {"kcal_per_100g": 300, "protein_g": 10.0, "fat_g": 8.0,  "carb_g": 25.0, "vitamins": "A, C",       "type": "bulk",      "avg_weight_g": 200},
    "roti":                {"kcal_per_100g": 270, "protein_g": 6.0,  "fat_g": 10.0, "carb_g": 40.0, "vitamins": "B1",         "type": "countable", "avg_weight_g": 100},
    "cherry_tomato":       {"kcal_per_100g": 18,  "protein_g": 0.9,  "fat_g": 0.2,  "carb_g": 3.9,  "vitamins": "C, K, A",    "type": "countable", "avg_weight_g": 10},
    "tomato":              {"kcal_per_100g": 18,  "protein_g": 0.9,  "fat_g": 0.2,  "carb_g": 3.9,  "vitamins": "C, K, A",    "type": "countable", "avg_weight_g": 120},
    "cucumber":            {"kcal_per_100g": 16,  "protein_g": 0.7,  "fat_g": 0.1,  "carb_g": 3.6,  "vitamins": "K, C",       "type": "countable", "avg_weight_g": 300},
    "cucumber_full":       {"kcal_per_100g": 16,  "protein_g": 0.7,  "fat_g": 0.1,  "carb_g": 3.6,  "vitamins": "K, C",       "type": "countable", "avg_weight_g": 300},
    "lettuce":             {"kcal_per_100g": 15,  "protein_g": 1.4,  "fat_g": 0.2,  "carb_g": 2.9,  "vitamins": "K, A",       "type": "bulk",      "avg_weight_g": 50},
    "cabbage":             {"kcal_per_100g": 25,  "protein_g": 1.3,  "fat_g": 0.1,  "carb_g": 6.0,  "vitamins": "K, C",       "type": "countable", "avg_weight_g": 500},
    "cabbage_sliced":      {"kcal_per_100g": 25,  "protein_g": 1.3,  "fat_g": 0.1,  "carb_g": 6.0,  "vitamins": "K, C",       "type": "bulk",      "avg_weight_g": 100},
    "cabbage_stirfried":   {"kcal_per_100g": 40,  "protein_g": 1.5,  "fat_g": 2.0,  "carb_g": 5.0,  "vitamins": "K, C",       "type": "bulk",      "avg_weight_g": 100},
    "spinach":             {"kcal_per_100g": 23,  "protein_g": 2.9,  "fat_g": 0.4,  "carb_g": 3.6,  "vitamins": "A, K, C",    "type": "bulk",      "avg_weight_g": 30},
    "brocolli":            {"kcal_per_100g": 34,  "protein_g": 2.8,  "fat_g": 0.4,  "carb_g": 7.0,  "vitamins": "C, K, A",    "type": "countable", "avg_weight_g": 91},
    "chinese_broccoli":    {"kcal_per_100g": 35,  "protein_g": 2.5,  "fat_g": 0.5,  "carb_g": 7.0,  "vitamins": "C, A, K",    "type": "countable", "avg_weight_g": 80},
    "cauliflower":         {"kcal_per_100g": 25,  "protein_g": 1.9,  "fat_g": 0.3,  "carb_g": 5.0,  "vitamins": "C, K",       "type": "countable", "avg_weight_g": 100},
    "baby_corn":           {"kcal_per_100g": 26,  "protein_g": 2.0,  "fat_g": 0.5,  "carb_g": 6.0,  "vitamins": "C, A",       "type": "countable", "avg_weight_g": 30},
    "sweetcorn":           {"kcal_per_100g": 86,  "protein_g": 3.2,  "fat_g": 1.2,  "carb_g": 19.0, "vitamins": "C, B1, B5",  "type": "bulk",      "avg_weight_g": 100},
    "corn":                {"kcal_per_100g": 96,  "protein_g": 3.4,  "fat_g": 1.5,  "carb_g": 21.0, "vitamins": "C, B1, B5",  "type": "bulk",      "avg_weight_g": 100},
    "peas":                {"kcal_per_100g": 81,  "protein_g": 5.4,  "fat_g": 0.4,  "carb_g": 14.0, "vitamins": "K, C, A",    "type": "bulk",      "avg_weight_g": 100},
    "yardlong_bean":       {"kcal_per_100g": 47,  "protein_g": 2.8,  "fat_g": 0.4,  "carb_g": 8.0,  "vitamins": "C, A",       "type": "countable", "avg_weight_g": 100},
    "bean_sprouts":        {"kcal_per_100g": 30,  "protein_g": 3.0,  "fat_g": 0.1,  "carb_g": 6.0,  "vitamins": "C, K",       "type": "bulk",      "avg_weight_g": 100},
    "bell_pepper":         {"kcal_per_100g": 31,  "protein_g": 1.0,  "fat_g": 0.3,  "carb_g": 6.0,  "vitamins": "C, A, B6",   "type": "countable", "avg_weight_g": 120},
    "bell_pepper_cuted":   {"kcal_per_100g": 31,  "protein_g": 1.0,  "fat_g": 0.3,  "carb_g": 6.0,  "vitamins": "C, A, B6",   "type": "countable", "avg_weight_g": 20},
    "capsicum":            {"kcal_per_100g": 31,  "protein_g": 1.0,  "fat_g": 0.3,  "carb_g": 6.0,  "vitamins": "C, A, B6",   "type": "countable", "avg_weight_g": 120},
    "chilli_pepper":       {"kcal_per_100g": 40,  "protein_g": 1.9,  "fat_g": 0.4,  "carb_g": 8.8,  "vitamins": "C, B6, A",   "type": "countable", "avg_weight_g": 45},
    "carrot":              {"kcal_per_100g": 41,  "protein_g": 0.9,  "fat_g": 0.2,  "carb_g": 10.0, "vitamins": "A, K",       "type": "countable", "avg_weight_g": 60},
    "carrot_ring":         {"kcal_per_100g": 41,  "protein_g": 0.9,  "fat_g": 0.2,  "carb_g": 10.0, "vitamins": "A, K",       "type": "countable", "avg_weight_g": 10},
    "courgette":           {"kcal_per_100g": 17,  "protein_g": 1.2,  "fat_g": 0.3,  "carb_g": 3.1,  "vitamins": "A, C",       "type": "countable", "avg_weight_g": 200},
    "cantaloupe":          {"kcal_per_100g": 34,  "protein_g": 0.8,  "fat_g": 0.2,  "carb_g": 8.0,  "vitamins": "C, A",       "type": "bulk",      "avg_weight_g": 160},
    "cantaloupe_pieces":   {"kcal_per_100g": 34,  "protein_g": 0.8,  "fat_g": 0.2,  "carb_g": 8.0,  "vitamins": "C, A",       "type": "bulk",      "avg_weight_g": 50},
    "coconut":             {"kcal_per_100g": 354, "protein_g": 3.3,  "fat_g": 33.5, "carb_g": 15.2, "vitamins": "Manganese",  "type": "countable", "avg_weight_g": 100},
    "dates":               {"kcal_per_100g": 282, "protein_g": 2.5,  "fat_g": 0.4,  "carb_g": 75.0, "vitamins": "B6, Iron",   "type": "countable", "avg_weight_g": 24},
    "dragon_fruit":        {"kcal_per_100g": 50,  "protein_g": 1.2,  "fat_g": 0.0,  "carb_g": 13.0, "vitamins": "C, Iron",    "type": "countable", "avg_weight_g": 600},
    "durian":              {"kcal_per_100g": 147, "protein_g": 1.5,  "fat_g": 5.3,  "carb_g": 27.0, "vitamins": "C, B1",      "type": "bulk",      "avg_weight_g": 100},
    "eggplant":            {"kcal_per_100g": 25,  "protein_g": 1.0,  "fat_g": 0.2,  "carb_g": 6.0,  "vitamins": "K, B5",      "type": "countable", "avg_weight_g": 300},
    "garlic":              {"kcal_per_100g": 149, "protein_g": 6.4,  "fat_g": 0.5,  "carb_g": 33.0, "vitamins": "C, B6",      "type": "countable", "avg_weight_g": 3},
    "ginger":              {"kcal_per_100g": 80,  "protein_g": 1.8,  "fat_g": 0.8,  "carb_g": 18.0, "vitamins": "B6",         "type": "bulk",      "avg_weight_g": 5},
    "grapes":              {"kcal_per_100g": 69,  "protein_g": 0.7,  "fat_g": 0.2,  "carb_g": 18.0, "vitamins": "K, C",       "type": "countable", "avg_weight_g": 5},
    "kiwi":                {"kcal_per_100g": 61,  "protein_g": 1.1,  "fat_g": 0.5,  "carb_g": 15.0, "vitamins": "C, K",       "type": "countable", "avg_weight_g": 76},
    "lemon":               {"kcal_per_100g": 29,  "protein_g": 1.1,  "fat_g": 0.3,  "carb_g": 9.0,  "vitamins": "C",          "type": "countable", "avg_weight_g": 65},
    "lychee":              {"kcal_per_100g": 66,  "protein_g": 0.8,  "fat_g": 0.4,  "carb_g": 17.0, "vitamins": "C",          "type": "countable", "avg_weight_g": 20},
    "mango":               {"kcal_per_100g": 60,  "protein_g": 0.8,  "fat_g": 0.4,  "carb_g": 15.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 200},
    "nectarine":           {"kcal_per_100g": 44,  "protein_g": 1.1,  "fat_g": 0.3,  "carb_g": 11.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 142},
    "olive":               {"kcal_per_100g": 115, "protein_g": 0.8,  "fat_g": 11.0, "carb_g": 6.0,  "vitamins": "E, Iron",    "type": "countable", "avg_weight_g": 5},
    "onion":               {"kcal_per_100g": 40,  "protein_g": 1.1,  "fat_g": 0.1,  "carb_g": 9.3,  "vitamins": "C, B6",      "type": "countable", "avg_weight_g": 110},
    "orange":              {"kcal_per_100g": 47,  "protein_g": 0.9,  "fat_g": 0.1,  "carb_g": 12.0, "vitamins": "C",          "type": "countable", "avg_weight_g": 130},
    "papaya":              {"kcal_per_100g": 43,  "protein_g": 0.5,  "fat_g": 0.3,  "carb_g": 11.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 130},
    "paprika":             {"kcal_per_100g": 315, "protein_g": 12.0, "fat_g": 13.0, "carb_g": 54.0, "vitamins": "A, E, B6",   "type": "bulk",      "avg_weight_g": 2},
    "passion":             {"kcal_per_100g": 97,  "protein_g": 2.2,  "fat_g": 0.7,  "carb_g": 23.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 18},
    "peach":               {"kcal_per_100g": 39,  "protein_g": 0.9,  "fat_g": 0.3,  "carb_g": 10.0, "vitamins": "C, A",       "type": "countable", "avg_weight_g": 150},
    "pear":                {"kcal_per_100g": 57,  "protein_g": 0.4,  "fat_g": 0.1,  "carb_g": 15.0, "vitamins": "C, K",       "type": "countable", "avg_weight_g": 178},
    "pepino":              {"kcal_per_100g": 27,  "protein_g": 0.6,  "fat_g": 0.1,  "carb_g": 6.0,  "vitamins": "C, A",       "type": "countable", "avg_weight_g": 250},
    "pineapple":           {"kcal_per_100g": 50,  "protein_g": 0.5,  "fat_g": 0.1,  "carb_g": 13.0, "vitamins": "C, Manganese", "type": "countable", "avg_weight_g": 905},
    "plum":                {"kcal_per_100g": 46,  "protein_g": 0.7,  "fat_g": 0.3,  "carb_g": 11.0, "vitamins": "C, K, A",    "type": "countable", "avg_weight_g": 66},
    "pomegranate":         {"kcal_per_100g": 83,  "protein_g": 1.7,  "fat_g": 1.2,  "carb_g": 19.0, "vitamins": "C, K",       "type": "countable", "avg_weight_g": 282},
    "potato":              {"kcal_per_100g": 77,  "protein_g": 2.0,  "fat_g": 0.1,  "carb_g": 17.0, "vitamins": "C, B6",      "type": "countable", "avg_weight_g": 213},
    "pumpkin":             {"kcal_per_100g": 26,  "protein_g": 1.0,  "fat_g": 0.1,  "carb_g": 7.0,  "vitamins": "A, C",       "type": "countable", "avg_weight_g": 923},
    "raddish":             {"kcal_per_100g": 16,  "protein_g": 0.7,  "fat_g": 0.1,  "carb_g": 3.4,  "vitamins": "C",          "type": "countable", "avg_weight_g": 58},
    "radish_ring":         {"kcal_per_100g": 16,  "protein_g": 0.7,  "fat_g": 0.1,  "carb_g": 3.4,  "vitamins": "C",          "type": "countable", "avg_weight_g": 10},
    "sugar_apple":         {"kcal_per_100g": 94,  "protein_g": 1.7,  "fat_g": 0.6,  "carb_g": 24.0, "vitamins": "C, B6",      "type": "countable", "avg_weight_g": 200},
    "sweetpotato":         {"kcal_per_100g": 86,  "protein_g": 1.6,  "fat_g": 0.1,  "carb_g": 20.0, "vitamins": "A, C, B6",   "type": "countable", "avg_weight_g": 130},
    "turnip":              {"kcal_per_100g": 28,  "protein_g": 0.9,  "fat_g": 0.1,  "carb_g": 6.0,  "vitamins": "C",          "type": "countable", "avg_weight_g": 98},
    "watermelon":          {"kcal_per_100g": 30,  "protein_g": 0.6,  "fat_g": 0.2,  "carb_g": 8.0,  "vitamins": "C, A",       "type": "countable", "avg_weight_g": 280},
    "avocado":             {"kcal_per_100g": 160, "protein_g": 2.0,  "fat_g": 15.0, "carb_g": 9.0,  "vitamins": "K, E, C",    "type": "countable", "avg_weight_g": 150},
    "avocado_cuted":       {"kcal_per_100g": 160, "protein_g": 2.0,  "fat_g": 15.0, "carb_g": 9.0,  "vitamins": "K, E, C",    "type": "countable", "avg_weight_g": 75},
    "apple":               {"kcal_per_100g": 52,  "protein_g": 0.3,  "fat_g": 0.2,  "carb_g": 14.0, "vitamins": "C",          "type": "countable", "avg_weight_g": 180},
    "apple_cuted":         {"kcal_per_100g": 52,  "protein_g": 0.3,  "fat_g": 0.2,  "carb_g": 14.0, "vitamins": "C",          "type": "countable", "avg_weight_g": 10},
    "coriander":           {"kcal_per_100g": 23,  "protein_g": 2.1,  "fat_g": 0.5,  "carb_g": 3.7,  "vitamins": "A, K, C",    "type": "bulk",      "avg_weight_g": 30},
    "donut": {"kcal_per_100g": 350, "protein_g": 5.0, "fat_g": 18.0, "carb_g": 41.0, "vitamins": "ไม่มีข้อมูลเฉพาะเจาะจง", "type": "snack", "avg_weight_g": 50},
    "french_fries": {"kcal_per_100g": 296, "protein_g": 3.0, "fat_g": 15.0, "carb_g": 37.0, "vitamins": "ไม่มีข้อมูลเฉพาะเจาะจง", "type": "side_dish", "avg_weight_g": 100}
}
FOOD_DB = {k.lower().replace("_", "").replace("-", ""): v for k, v in food_metadata.items()}
REFERENCE_OBJECTS = {
    "10": {"width_mm": 26, "height_mm": 26},
    "5": {"width_mm": 24, "height_mm": 24},
    "1": {"width_mm": 20, "height_mm": 20},
    "creditcard": {"width_mm": 85.6, "height_mm": 53.98},
}

MODELS_DIR = "models"
COIN_MODEL_NAME = "coin_detector.pt" 


@st.cache_resource
def load_models():
    st.info("Loading models... This may take a moment.")
    coin_model = None
    food_models = []
    
    try:
        if not os.path.exists(MODELS_DIR):
            return None, [], f"Error: '{MODELS_DIR}' directory not found. Please create it and add your .pt model files."
            
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
        if not model_files:
            return None, [], f"Error: No .pt models found in '{MODELS_DIR}'."
            
        coin_model_path = os.path.join(MODELS_DIR, COIN_MODEL_NAME)
        if os.path.exists(coin_model_path):
            coin_model = YOLO(coin_model_path)
            
        for model_file in model_files:
            if model_file != COIN_MODEL_NAME:
                model_path = os.path.join(MODELS_DIR, model_file)
                food_models.append(YOLO(model_path))
                
        if not food_models:
             return coin_model, food_models, "Warning: Food model not found. Only reference object detection is possible."
             
        return coin_model, food_models, "Models loaded successfully!"
        
    except Exception as e:
        return None, [], f"Error loading models: {e}"


def estimate_food_weight(x1, y1, x2, y2, pixels_per_mm, food_info):
    if not pixels_per_mm or "density_g_per_cm3" not in food_info: return None 
    
    DENSITY = food_info["density_g_per_cm3"] 
    SHAPE = food_info.get("shape", "irregular")
    
    width_mm = (x2 - x1) / pixels_per_mm
    height_mm = (y2 - y1) / pixels_per_mm
    width_cm, height_cm = width_mm / 10, height_mm / 10
    
    estimated_volume_cm3 = 0
    if SHAPE == "sphere":
        diameter_cm = (width_cm + height_cm) / 2
        radius_cm = diameter_cm / 2
        estimated_volume_cm3 = (4/3) * np.pi * (radius_cm**3) 
    elif SHAPE == "cylinder":
        diameter_cm = min(width_cm, height_cm)
        radius_cm = diameter_cm / 2 
        height_of_cylinder_cm = max(width_cm, height_cm)
        estimated_volume_cm3 = np.pi * (radius_cm**2) * height_of_cylinder_cm 
    else: 
        area_cm2 = width_cm * height_cm
        
        assumed_thickness_cm = ((width_cm + height_cm) / 2) * 0.35 
        estimated_volume_cm3 = area_cm2 * assumed_thickness_cm 
        
    if estimated_volume_cm3 > 0:
        return estimated_volume_cm3 * DENSITY
    return None


def process_image(frame, coin_model, food_models, high_accuracy_mode):
    if not food_models and not coin_model:
        return frame, 0, ["**ERROR:** Models not loaded. Please check the 'models' folder."]

    display_frame = frame.copy()
    total_calories = 0
    pixels_per_mm = None
    
    def normalize_name(name):
        return name.lower().replace("_", "").replace("-", "")

    
    if high_accuracy_mode and coin_model:
        coin_results = coin_model(frame, conf=0.1, verbose=False)
        for r in coin_results:
            for box in r.boxes:
                class_name_raw = coin_model.names[int(box.cls[0])]
                class_name = normalize_name(class_name_raw)
                if class_name in REFERENCE_OBJECTS:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    ref_obj_info = REFERENCE_OBJECTS[class_name]
                    bbox_width_pixels = x2 - x1
                    pixels_per_mm = bbox_width_pixels / ref_obj_info["width_mm"]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display_frame, f"Ref: {class_name_raw} ({pixels_per_mm:.2f} P/mm)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    break
            if pixels_per_mm: break

    
    raw_boxes, raw_scores, all_detections_for_nms = [], [], []
    for model in food_models:
        food_results = model(frame, verbose=False)
        for r in food_results:
            for box in r.boxes:
                
                if box.xyxy.numel() > 0:
                    x1, y1, x2, y2 = box.xyxy[0]; w, h = x2 - x1, y2 - y1
                    
                    raw_boxes.append([int(x1), int(y1), int(w), int(h)]) 
                    raw_scores.append(float(box.conf[0]))
                    all_detections_for_nms.append(box)
    
    indices = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, 0.45, 0.5) if raw_boxes else []

   
    aggregated_detections = {}
    if len(indices) > 0:
        for i in indices.flatten():
            box = all_detections_for_nms[i]
            class_id = int(box.cls[0])
            
            class_name_raw = food_models[0].names[class_id]
            
            if class_name_raw not in aggregated_detections:
                aggregated_detections[class_name_raw] = []
            aggregated_detections[class_name_raw].append(box)

  
    detected_items_info = []
    
    
    color_map = {
        "meat": (0, 165, 255), "seafood": (255, 105, 65), "noodle": (0, 255, 255), 
        "vegetable": (0, 200, 0), "fruit": (150, 50, 200), "egg": (0, 200, 255), "other": (200, 200, 200)
    }
    
    def get_color_by_type(food_name):
        if "pork" in food_name or "chicken" in food_name or "ham" in food_name or "sausage" in food_name or "meat" in food_name:
            return color_map["meat"]
        if "shrimp" in food_name or "crab" in food_name or "fish" in food_name or "squid" in food_name or "clam" in food_name or "opyster" in food_name:
            return color_map["seafood"]
        if "noodle" in food_name or "pad_thai" in food_name or "roti" in food_name:
            return color_map["noodle"]
        if "vegetable" in food_name or "spinach" in food_name or "brocolli" in food_name or "cabbage" in food_name or "lettuce" in food_name:
            return color_map["vegetable"]
        if "tomato" in food_name or "cucumber" in food_name or "carrot" in food_name or "bell_pepper" in food_name:
             return color_map["vegetable"]
        if "apple" in food_name or "mango" in food_name or "watermelon" in food_name or "kiwi" in food_name:
             return color_map["fruit"]
        if "egg" in food_name or "omelette" in food_name:
            return color_map["egg"]
        return color_map["other"]


    for class_name_raw, boxes in aggregated_detections.items():
        class_name = normalize_name(class_name_raw)
        if class_name in REFERENCE_OBJECTS: continue

        food_info = FOOD_DB.get(class_name)
        if not food_info:
            detected_items_info.append(f"**{class_name_raw}**: Food not in DB. Could not calculate.")
            continue

        final_weight_g = 0
        food_type = food_info.get("type", "countable")
        base_weight = food_info.get("avg_weight_g", 0)
        box_color = get_color_by_type(class_name)
        
      
        if high_accuracy_mode and pixels_per_mm:
            estimated_weight_g_list = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                estimated_weight = estimate_food_weight(x1, y1, x2, y2, pixels_per_mm, food_info)
                if estimated_weight is not None:
                    estimated_weight_g_list.append(estimated_weight)
            
            if estimated_weight_g_list:
                final_weight_g = sum(estimated_weight_g_list)
            else:
                final_weight_g = base_weight * len(boxes) if food_type == "countable" else base_weight
        else: 
            if food_type == "bulk":
                final_weight_g = base_weight
            else: 
                final_weight_g = base_weight
                if len(boxes) > 1 and base_weight > 0:
                    
                    final_weight_g += (len(boxes) - 1) * (base_weight * 0.25)
        
      
        if final_weight_g > 0:
            calories = (final_weight_g / 100) * food_info["kcal_per_100g"]
            total_calories += calories
            count_text = f" ({len(boxes)} pcs)" if len(boxes) > 1 and food_type == "countable" else ""
            info_text = f"**{class_name_raw}{count_text}** (~{final_weight_g:.0f}g): **{calories:.1f} kcal**"
            detected_items_info.append(info_text)
        else:
            detected_items_info.append(f"**{class_name_raw}**: No weight info (Weight 0g)")

       
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            
            text_label = f"{class_name_raw}"
            (text_width, text_height), baseline = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1)
            
            cv2.putText(display_frame, text_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
   
    warning_string = ""
    if high_accuracy_mode and not pixels_per_mm and len(indices) > 0:
        warning_string = "**คำเตือน:** เปิดโหมด High Accuracy แต่ไม่พบวัตถุอ้างอิง (เหรียญ/บัตรเครดิต) การคำนวณน้ำหนักอาจคลาดเคลื่อน\n\n"
    
    info_markdown = warning_string + "\n".join(detected_items_info)

    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    
    return display_frame_rgb, total_calories, info_markdown


def main():
    st.set_page_config(page_title="YOLOv8 Calorie Detector", layout="wide")
    st.title("🍽️ AI Calorie Detection Web App")
    st.markdown("##### 🔍 Select a sample image below to see the result instantly, or upload your own photo in the sidebar.")
    
    
    TEST_IMAGE_DIR = "testpic"
    uploaded_file = None
    
    
    if os.path.exists(TEST_IMAGE_DIR):
        all_files = os.listdir(TEST_IMAGE_DIR)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))] 
    else:
        image_files = []
        st.error(f"Error: '{TEST_IMAGE_DIR}' folder not found on the server.")
        
    
    if image_files:
        st.subheader("Example Images (Click to Select)")
        
       
        cols = st.columns(min(len(image_files), 5)) 
        
        
        if 'selected_sample_file' not in st.session_state:
            st.session_state.selected_sample_file = None
            
       
        for i, file_name in enumerate(image_files):
            with cols[i % 5]: 
                file_path = os.path.join(TEST_IMAGE_DIR, file_name)
                
                
                try:
                    
                    sample_image = Image.open(file_path)
                    st.image(sample_image, caption=file_name, width=100)
                except Exception as e:
                    st.error(f"Cannot display {file_name}")
                    
                
                if st.button(f"Select {i+1}", key=f"select_btn_{i}"):
                    st.session_state.selected_sample_file = file_path
                    
                    st.rerun()

    
    if st.session_state.selected_sample_file:
        selected_file_path = st.session_state.selected_sample_file
        
      
        with open(selected_file_path, "rb") as f:
            image_data = f.read()
        
        
        uploaded_file = BytesIO(image_data)
        uploaded_file.name = os.path.basename(selected_file_path) 

        st.success(f"Selected sample image: **{uploaded_file.name}**")
        
    
    with st.sidebar:
        st.header("⚙️ Settings")
        high_accuracy_mode = st.checkbox("High Accuracy Mode (ใช้การตรวจจับเหรียญ/บัตร)", value=True)
        st.markdown("---")
        
        st.header("Or Upload Your Own Image")
        
        user_uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ", type=['jpg', 'jpeg', 'png', 'jfif'])
        
        if user_uploaded_file is not None:
             uploaded_file = user_uploaded_file 

        st.markdown("---")
        st.subheader("Model Status")
        coin_model, food_models, model_status = load_models()
        st.caption(model_status)
<<<<<<< Updated upstream
        
=======


   
>>>>>>> Stashed changes
    col1, col2 = st.columns([1, 2])
    
    with col1:
        total_cal_placeholder = st.empty()
        total_cal_placeholder.metric(label="Total Estimated Calories", value="0.0 kcal", delta="Waiting for image...")

    if uploaded_file is not None:
        
        try:
            
            image_pil = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image_pil)
            frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            with st.spinner('Running YOLOv8 detection and calculation...'):
                processed_image, total_calories, info_markdown = process_image(frame, coin_model, food_models, high_accuracy_mode)
            
           
            total_cal_placeholder.metric(label="Total Estimated Calories", value=f"{total_calories:.1f} kcal", delta="Estimated")
            
            
            st.image(processed_image, caption=f'Detection Results for {uploaded_file.name}', use_column_width=True)
            
          
            st.subheader("Detected Items and Calorie Breakdown")
            st.markdown(info_markdown)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please ensure you have resolved the dependency issues and have valid models.")
    else:
        st.info("👈 Please select a sample image above or upload your own image from the sidebar to begin.")

if __name__ == '__main__':
    main()
