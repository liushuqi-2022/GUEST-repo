import PIL
from appium import webdriver
from appium.webdriver.common.mobileby import MobileBy
import layout_tree as LayoutTree
import time
import os, csv
from appium.webdriver.common.touch_action import TouchAction
import pickle
import sys
import PIL.Image
import psutil
import networkx as nx
from collections import Counter
from test_generator_auto import DestEvent

current_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir_path, '..', 'model_generation'))

from entities import IR_Model
from pathlib import Path
import pandas as pd
import json

class Explorer:
    def __init__(self, desiredCapabilities):
        # replace with you own desired capabilities for Appium
        self.desiredCapabilities = desiredCapabilities
        # make sure too change to port your Appium server is listening on
        d = webdriver.Remote('http://localhost:4723/wd/hub', self.desiredCapabilities)
        assert d is not None
        self.driver = d

    def execute_test(self, test_file):
        test = pickle.load(open(test_file, 'rb'))
        for event in test:
            if type(event) is list:
                for self_event in event:
                    self.execute_event(self_event)
            else:
                self.execute_event(event)

    def get_current_widgetIR(self, event, annotation_df):
        crop_path = event.crop_screenshot_path
        row_found = annotation_df.loc[annotation_df['filepath'] == crop_path]
        if len(row_found) == 0:
            image = PIL.Image.open(event.crop_screenshot_path)
            image.show()
            widgetIR = input('type widget IR that is about to trigger\n')
            annotation_df = annotation_df.append({'filepath' : crop_path, 'IR': widgetIR}, ignore_index = True)
            return widgetIR, annotation_df
        elif len(row_found) == 1:
            if pd.isna(row_found['IR'].values[0]):
                image = PIL.Image.open(event.crop_screenshot_path)
                image.show()
                widgetIR = input('type widget IR that is about to trigger\n')
                annotation_df = annotation_df.append({'filepath' : crop_path, 'IR': widgetIR}, ignore_index = True)
                return widgetIR, annotation_df
            return row_found['IR'].values[0], annotation_df
        else:
            raise ValueError('row found is > 1 when getting widgetIR, check', event.crop_screenshot_path)

    def get_current_screenIR(self, event, annotation_df):
        screenshot_path = event.state_screenshot_path
        row_found = annotation_df.loc[annotation_df['filepath'] == screenshot_path]
        if len(row_found) == 0:
            image = PIL.Image.open(event.state_screenshot_path)
            image.show()
            current_screenIR = input('type current screen IR shown in the screenshot\n')
            annotation_df = annotation_df.append({'filepath' : screenshot_path, 'IR': current_screenIR}, ignore_index = True)
            return current_screenIR, annotation_df
        elif len(row_found) == 1:
            if pd.isna(row_found['IR'].values[0]):
                image = PIL.Image.open(event.state_screenshot_path)
                image.show()
                current_screenIR = input('type current screen IR shown in the screenshot\n')
                annotation_df = annotation_df.append({'filepath' : screenshot_path, 'IR': current_screenIR}, ignore_index = True)
                return current_screenIR, annotation_df
            return row_found['IR'].values[0], annotation_df
        else:
            raise ValueError('row found is > 1 when getting screenIR, check', event.state_screenshot_path)

    def execute_test_and_generate_linear_model(self, test_file):
        test = pickle.load(open(test_file, 'rb'))
        linear_model = []
        dynamic_annotation_file = os.path.join(Path(test_file).parent.parent.parent.parent.absolute(), 'dynamic_annotations.csv')
        if not os.path.exists(dynamic_annotation_file):
            headers = ['filepath', 'IR']
            with open(dynamic_annotation_file, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(headers)
        annotation_df = pd.read_csv(dynamic_annotation_file)
        for event in test:
            if type(event) is list:
                for self_event in event:
                    current_screenIR, annotation_df = self.get_current_screenIR(self_event, annotation_df)
                    action = self_event.action
                    if 'swipe' in action:
                        swipe_direction = event.action.split('-')[1]
                        linear_model.append({'state': current_screenIR, 'transition': swipe_direction})
                    else:
                        widgetIR, annotation_df = self.get_current_widgetIR(self_event, annotation_df)
                        transition_name = widgetIR + '#' + action
                        linear_model.append({'state': current_screenIR, 'transition': transition_name})
                    self.execute_event(self_event)
            else:
                current_screenIR, annotation_df = self.get_current_screenIR(event, annotation_df)
                action = event.action
                if 'swipe' in action:
                    swipe_direction = event.action.split('-')[1]
                    linear_model.append({'state': current_screenIR, 'transition': swipe_direction})
                else:
                    widgetIR, annotation_df = self.get_current_widgetIR(event, annotation_df)
                    transition_name = widgetIR + '#' + action
                    linear_model.append({'state': current_screenIR, 'transition': transition_name})
                self.execute_event(event)
        annotation_df.to_csv(dynamic_annotation_file, index=False)
        with open(test_file.replace('.pickle', '-linear.json'), 'w') as output:
            json.dump(linear_model, output)

    def is_final_trigger(self, usage_model, trigger, source):
        if len(usage_model.machine.get_transitions(trigger=trigger, source=source, dest='end')) == 0:
            return False
        return True

    def find_possible_next_actions_new(self, current_state, matching_screenIR, triggers, match_scores, usage_model, operable_widgets, similardict, text_sim_bert, usage_name, nodes):
        SUGGESTION_CNT = 10
        top_actions = []
        heuristic_actions = []
        tops = set()
        heuristics = set()
        if matching_screenIR in ["category", "items"]:
            if matching_screenIR == "category":
                if "category_i#click" not in triggers:
                    triggers.append("category_i#click")
            else:
                if "item_i#click" not in triggers:
                    triggers.append("item_i#click")

        for trigger in triggers:
            if trigger == 'self':
                raise ValueError('self trigger should be removed already, check triggers of ', matching_screenIR)
            isEnd = self.is_final_trigger(usage_model, trigger=trigger, source=matching_screenIR)

            widgetIR = trigger.split('#')[0]
            top_matches, heuristic_based_matches, matchsimilar = current_state.find_widget_candidates(
                        widgetIR, matching_screenIR, operable_widgets, similardict, text_sim_bert, usage_name)

            if len(heuristic_based_matches) > 0:
                for match in heuristic_based_matches:
                    action = trigger.split('#')[-1]
                    if match.get_element_type().split('.')[-1] == "EditText":
                        action = "send_keys"
                    if match.get_exec_id_val() not in heuristics and match.get_exec_id_val() != "":
                        heuristic_actions.append(DestEvent(class0=match.attributes['class'], action=action,
                                                           exec_id_type=match.get_exec_id_type(),
                                                           exec_id_val=match.get_exec_id_val(), text=match.get_text(),
                                                           text_input='',
                                                           isEnd=isEnd,
                                                           crop_screenshot_path=match.path_to_screenshot,
                                                           state_screenshot_path=current_state.screenshot_path,
                                                           matched_trigger=trigger))
                        heuristics.add(match.get_exec_id_val())
                        match_scores[match.get_exec_id_val()] = matchsimilar[match]

                        if match.get_exec_id_val() in tops:
                            tops.remove(match.get_exec_id_val())

            if len(top_matches) > 0:
                for match in top_matches:
                    if match.get_exec_id_val() == "":
                        for node in nodes:
                            if 'rotation' in node.attributes.keys():
                                continue
                            if node.attributes['text'] == '':
                                if 'resource-id' not in node.attributes.keys() or node.attributes['resource-id'] == '':
                                    continue
                            if node.attributes['bounds'] == match.attributes['bounds']:
                                match0 = node
                                for attr in ['text', 'resource-id']:
                                    if attr in node.attributes.keys():
                                        node.add_data(attr, node.attributes[attr])
                                        if node.attributes[attr] != '':
                                            node.add_exec_identifier(attr, node.attributes[attr])
                                break
                            else:
                                if self.explorer.is_inbounds(match.attributes['bounds'], node.get_middle_point()):
                                    if node.attributes['text'] != '' and len(node.attributes['text'].split()) != 1:
                                        match0 = node
                                        for attr in ['text', 'resource-id']:
                                            if attr in node.attributes.keys():
                                                node.add_data(attr, node.attributes[attr])
                                                if node.attributes[attr] != '':
                                                    node.add_exec_identifier(attr, node.attributes[attr])
                                        break

                    action = trigger.split('#')[-1]
                    if match.get_element_type().split('.')[-1] == "EditText":
                        action = "send_keys"

                    if match.get_exec_id_val() != "":
                        if (match.get_exec_id_val() not in tops) and (match.get_exec_id_val() not in heuristics):
                            top_actions.append(DestEvent(class0=match.attributes['class'], action=action,
                                                         exec_id_type=match.get_exec_id_type(),
                                                         exec_id_val=match.get_exec_id_val(), text=match.get_text(),
                                                         text_input='', isEnd=isEnd,
                                                         crop_screenshot_path=match.path_to_screenshot,
                                                         state_screenshot_path=current_state.screenshot_path,
                                                         matched_trigger=trigger))

                            tops.add(match.get_exec_id_val())
                            match_scores[match.get_exec_id_val()] = matchsimilar[match]
                        if match.get_exec_id_val() in tops:
                            for i in top_actions:
                                if trigger != i.matched_trigger and match.get_exec_id_val() == i.exec_id_val:
                                    top_actions.append(DestEvent(class0=match.attributes['class'], action=action,
                                                                 exec_id_type=match.get_exec_id_type(),
                                                                 exec_id_val=match.get_exec_id_val(),
                                                                 text=match.get_text(), text_input='', isEnd=isEnd,
                                                                 crop_screenshot_path=match.path_to_screenshot,
                                                                 state_screenshot_path=current_state.screenshot_path,
                                                                 matched_trigger=trigger))
                                    break
                                else:
                                    if trigger == "category_i#click" and match.attributes['text'] == '':
                                        top_actions.append(DestEvent(class0=match.attributes['class'], action=action,
                                                                     exec_id_type=match.get_exec_id_type(),
                                                                     exec_id_val=match.get_exec_id_val(),
                                                                     text=match.get_text(), text_input='', isEnd=isEnd,
                                                                     crop_screenshot_path=match.path_to_screenshot,
                                                                     state_screenshot_path=current_state.screenshot_path,
                                                                     matched_trigger=trigger))
                                        break

                    else:
                        top_actions.append(DestEvent(class0=match0.attributes['class'], action=action,
                                                     exec_id_type=match0.get_exec_id_type(),
                                                     exec_id_val=match0.get_exec_id_val(), text=match0.get_text(),
                                                     text_input='', isEnd=isEnd,
                                                     crop_screenshot_path=match.path_to_screenshot,
                                                     state_screenshot_path=current_state.screenshot_path,
                                                     matched_trigger=trigger))
                        tops.add(match0.get_exec_id_val())
                        match_scores[match0.get_exec_id_val()] = matchsimilar[match]

        if len(heuristic_actions) >= SUGGESTION_CNT:
            return heuristic_actions[:SUGGESTION_CNT]
        elif len(heuristic_actions) + len(top_actions) > SUGGESTION_CNT:
            chosen_top_actions = top_actions[:(SUGGESTION_CNT - len(heuristic_actions))]
            return heuristic_actions + chosen_top_actions
        else:
            return heuristic_actions + top_actions

    def getcandidates_screenIR(self, digraph, pretrigger, step_index, prescIR):
        #
        successors = list(digraph.successors(prescIR))
        successors_successors = []
        for s in successors:
            successors_successors.extend(digraph.successors(s))
        successors.extend(successors_successors)

        trigger_sr = pretrigger.split('#')
        if trigger_sr[0].startswith('to'):
            if trigger_sr[0][3:] in digraph.nodes:
                successors.append(trigger_sr[0][3:])
        if 'popup' in digraph.nodes and 'popup' not in successors:
            successors.append('popup')
        if step_index > 0:
            successors.append(prescIR)
        merged_list = list(set(successors))
        return merged_list

    def extract_state(self, output_dir, test_num):
        layout = LayoutTree.LayoutTree(self.driver, output_dir)
        activity = self.driver.current_activity
        curr_state = layout.extract_state()
        curr_state.set_activity(activity)
        for element in curr_state.nodes:
            if element.interactable:
                if 'content-desc' in element.attributes.keys():
                    element.add_data('content-desc', element.attributes['content-desc'])
                    if element.attributes['content-desc'] != '':
                        element.add_exec_identifier('accessibility-id', element.attributes['content-desc'])

                if 'id' in element.attributes.keys():
                    element.add_data('id', element.attributes['id'])
                    if element.attributes['id'] != '':
                        element.add_exec_identifier('id', element.attributes['id'])

                if 'resource-id' in element.attributes.keys():
                    element.add_data('resource-id', element.attributes['resource-id'])
                    if element.attributes['resource-id'] != '':
                        element.add_exec_identifier('resource-id', element.attributes['resource-id'])

                if 'text' in element.attributes.keys():
                    element.add_data('text', element.attributes['text'])
                    element.add_exec_identifier('text', element.attributes['text'])

        if not os.path.isdir(os.path.join(output_dir, 'screenshots')):
            os.makedirs(os.path.join(output_dir, 'screenshots'))
        screenshot_path = os.path.join(output_dir, 'screenshots', str(test_num) + '-' + str(self.screenshot_idx) + '.png')
        self.driver.save_screenshot(screenshot_path)
        xml_path = os.path.join(output_dir, 'screenshots', str(test_num) + '-' + str(self.screenshot_idx) + '.xml')
        with open(xml_path, "w", encoding='utf-8') as file:
            file.write(self.driver.page_source)
        curr_state.add_screenshot_path(screenshot_path)
        curr_state.add_UIXML_path(xml_path)
        self.screenshot_idx += 1
        return curr_state

    def execute_swipe(self, direction):
        # Get screen dimensions
        screen_dimensions = self.driver.get_window_size()
        if direction == 'up':
            # Set co-ordinate X according to the element you want to scroll on.
            location_x = screen_dimensions["width"] * 0.5
            # Set co-ordinate start Y and end Y according to the scroll driection up or down
            location_start_y = screen_dimensions["height"] * 0.6
            location_end_y = screen_dimensions["height"] * 0.3
            # Perform vertical scroll gesture using TouchAction API.
            TouchAction(self.driver).press(x=location_x, y=location_start_y).wait(1000)\
                .move_to(x=location_x, y=location_end_y).release().perform()
        if direction == 'down':
            # Set co-ordinate X according to the element you want to scroll on.
            location_x = screen_dimensions["width"] * 0.5
            # Set co-ordinate start Y and end Y according to the scroll driection up or down
            location_start_y = screen_dimensions["height"] * 0.3
            location_end_y = screen_dimensions["height"] * 0.6
            # Perform vertical scroll gesture using TouchAction API.
            TouchAction(self.driver).press(x=location_x, y=location_start_y).wait(1000) \
                .move_to(x=location_x, y=location_end_y).release().perform()
        if direction == 'left':
            # Set co-ordinate start X and end X according
            location_start_x = screen_dimensions["width"] * 0.8
            location_end_x = screen_dimensions["width"] * 0.2
            # Set co-ordinate Y according to the element you want to swipe on.
            location_y = screen_dimensions["height"] * 0.5
            # Perform swipe gesture using TouchAction API.
            TouchAction(self.driver).press(x=location_start_x, y=location_y).wait(1000) \
                .move_to(x=location_end_x, y=location_y).release().perform()
        if direction == 'right':
            # Set co-ordinate start X and end X according
            location_start_x = screen_dimensions["width"] * 0.2
            location_end_x = screen_dimensions["width"] * 0.8
            # Set co-ordinate Y according to the element you want to swipe on.
            location_y = screen_dimensions["height"] * 0.5
            # Perform swipe gesture using TouchAction API.
            TouchAction(self.driver).press(x=location_start_x, y=location_y).wait(1000) \
                .move_to(x=location_end_x, y=location_y).release().perform()

    def find_hub_node(self, digraph):
        degree_centrality = nx.degree_centrality(digraph)
        degree_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
        closeness_centrality = nx.closeness_centrality(digraph)
        katz_centrality = nx.katz_centrality(digraph)
        harmonic_centrality = nx.harmonic_centrality(digraph)
        Com_centrality = dict(
            Counter(degree_centrality) + Counter(closeness_centrality) + Counter(katz_centrality) + Counter(
                harmonic_centrality))
        FCom_centrality = sorted(Com_centrality.items(), key=lambda item: item[1], reverse=True)
        ranked_screens = [x for x, _ in FCom_centrality]
        return ranked_screens

    def intimacy_new(self, next_screenir, end_nodes, hub_nodes, DG, prescIRs,hub_done):
        scoretotal = 0.01
        for i in next_screenir:
            if i in end_nodes or i in hub_nodes:
                inscore = 1
            else:
                if hub_done:
                    inscore = self.intimacy(i, end_nodes, DG, prescIRs)
                else:
                    inscore = self.intimacy(i, hub_nodes, DG, prescIRs)
            scoretotal += inscore
        return scoretotal / len(next_screenir)

    def intimacy(self, top_i, nodes, digraph, prescIRs):
        intotal1 = 0
        for j in nodes:
            unique_single_paths = set(tuple(path) for path in nx.all_simple_paths(digraph, top_i, j))
            total = 0
            pathnumber = 0
            jump = 0
            for path in unique_single_paths:
                for pathi in path:
                    if pathi in prescIRs:
                        jump = 1
                if jump == 1:
                    jump = 0
                    continue
                total = total + len(path) - 1
                pathnumber = pathnumber + 1
            if pathnumber == 0:
                in1 = 0
            else:
                in1 = float(pathnumber) / float((total / pathnumber) + 1)
            intotal1 += in1
        return intotal1

    def is_inbounds(self, Recy_bounds, midpoint):
        points = Recy_bounds.split("][")
        boundslist = [int(points[0].split(",")[0][1:]), int(points[0].split(",")[1]), int(points[1].split(",")[0]),
                      int(points[1].split(",")[1][:-1])]
        # [0,210][1080,1103]'->[0, 210, 1080, 1103]
        x = midpoint[0]
        y = midpoint[1]
        if boundslist[0] <= x <= boundslist[2] and boundslist[1] <= y <= boundslist[3]:
            return True
        else:
            return False

    def adjust_events(self, next_event_list, rank_type):
        new_next_event_list = next_event_list.copy()
        val = []
        index_list = []
        for i, event in enumerate(next_event_list):
            if rank_type == 'exec_id_val':
                if event.exec_id_val not in val:
                    val.append(event.exec_id_val)
                else:
                    count = val.count(event.exec_id_val)
                    if count < 2:
                        val.append(event.exec_id_val)
                    else:
                        index_list.append(i)

            if rank_type == 'matched_trigger':
                if event.matched_trigger not in val:
                    val.append(event.matched_trigger)
                else:
                    count = val.count(event.matched_trigger)
                    if count < 2:
                        val.append(event.matched_trigger)
                    else:
                        index_list.append(i)

        if index_list:
            for index_i in sorted(index_list, reverse=True):
                del new_next_event_list[index_i]
            for index_i in index_list:
                new_next_event_list.append(next_event_list[index_i])

        next_event_list = new_next_event_list
        return next_event_list

    def generate_user_input(self, next_event_list, current_state, match_scores):
        has_input_text = False
        for next_event in next_event_list:
            if type(next_event) == list:
                for e in next_event:
                    if e.action == "send_keys":
                        has_input_text = True
        if not has_input_text:
            self_actions = []
            print('generating user inputs...')
            for element in current_state.nodes:
                if element.interactable and 'EditText' in element.get_element_type():
                    image = PIL.Image.open(element.path_to_screenshot)
                    image.show()
                    adopt = input('Do you want to trigger this event? y or n?')
                    if adopt == 'y':
                        user_input = input('please enter your input for element that was just opened\n')

                        self_actions.append(
                            DestEvent(class0=element.attributes['class'], action='send_keys',
                                      exec_id_type=element.get_exec_id_type(),
                                      exec_id_val=element.get_exec_id_val(), text=element.get_text(),
                                      text_input=user_input, isEnd=False,
                                      crop_screenshot_path=element.path_to_screenshot,
                                      state_screenshot_path=current_state.screenshot_path))

                        match_scores[element.get_exec_id_val()] = 0.9
                        for proc in psutil.process_iter():
                            if proc.name() == 'Preview':
                                proc.kill()
                    else:
                        continue
            next_event_list = [self_actions] + next_event_list
            return next_event_list

    def execute_event(self, target_event, class0=None):
        # return the name of next state (will end test generation is it's 'end')
        element = None
        already_clicked = False
        actions = TouchAction(self.driver)

        if target_event.exec_id_type == "accessibility-id":
            time.sleep(3)
            element = self.driver.find_element_by_accessibility_id(target_event.exec_id_val)

        if target_event.exec_id_type == "xPath":
            xpath = '//' + class0 + '[@text='' + event.text + '']'
            time.sleep(3)
            element = self.driver.find_element(MobileBy.XPATH, xpath)

        if target_event.exec_id_type == "resource-id":
            time.sleep(3)
            element = self.driver.find_element_by_id(target_event.exec_id_val)

        if target_event.exec_id_type == "text":
            if class0 is None:
                class0 = target_event.class0
            time.sleep(3)
            xpath = '//' + class0 + '[@text="' + target_event.exec_id_val + '"]'
            element = self.driver.find_element(MobileBy.XPATH, xpath)

        if 'swipe' in target_event.action:
            time.sleep(3)
            swipe_direction = target_event.action.split('-')[1]
            self.execute_swipe(swipe_direction)

        if target_event.action == 'long':
            time.sleep(3)
            already_clicked = True
            actions.long_press(element).release().perform()

        if target_event.action == "send_keys":
            time.sleep(3)
            element.click()
            already_clicked = True
            time.sleep(1)
            element.send_keys(target_event.text_input)

        if target_event.action == "send_keys_enter":
            time.sleep(3)
            element.click()
            already_clicked = True
            time.sleep(3)
            element.send_keys(target_event.text_input)
            self.driver.press_keycode(66)

        if not already_clicked and target_event.action == 'click':
            element.click()

if __name__ == "__main__":
    desiredCapabilities = {
        "platformName": "Android",
        "deviceName": "emulator-5554",
        "newCommandTimeout": 10000,
        "appPackage": "com.foxnews.android",
        "appActivity": "com.foxnews.android.corenav.StartActivity"
    }
    explorer = Explorer(desiredCapabilities)
    event = {}
    # explorer.execute_event(event)
    time.sleep(3)
    element = explorer.driver.find_element_by_id(event['exec_id_val'])
    element.click()





