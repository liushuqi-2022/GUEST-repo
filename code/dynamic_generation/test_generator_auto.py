import sys
import PIL
import psutil as psutil
import os
import explorer
import time
import json
import os.path
import pickle
import glob
import pandas as pd
import networkx as nx
import nltk
from sentence_transformers import SentenceTransformer

current_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir_path, '..', 'model_generation'))
sys.path.insert(0, os.path.join(current_dir_path, 'similarity_calculate'))
sys.path.insert(0, os.path.join(current_dir_path, '..'))

from global_config import *
from bert_similarity_calc import SimilarityCalculator_BERT
from App_Config import *
from StrUtil import StrUtil

os.environ['TOKENIZERS_PARALLELISM'] = "False"


class TestCase:
    def __init__(self, test_folder_path):
        self.test_folder_path = test_folder_path
        self.events = []

    def add_event(self, action, state, image_path, text, text_input):
        event = Event(action, state, image_path, text, text_input)
        self.events.append(event)

    def print_test_case(self):
        for event in self.events:
            if event.action == "oracle-text_input":
                print("--------------------")
                print("state: " + event.state)
                print("action: " + event.action)
                print("oracle text input: " + event.text_input)
            else:
                print("--------------------")
                print("state: " + event.state)
                print("action: " + event.action)
                print("image_path: " + event.image_path)
                print(event.text)


class Event:
    def __init__(self, action, state, image_path, text, text_input):
        self.action = action
        self.state = state
        self.image_path = image_path
        self.text = text
        self.text_input = text_input


class DestEvent:
    def __init__(self, class0, action, exec_id_type, exec_id_val, text, text_input, isEnd, crop_screenshot_path,
                 state_screenshot_path, matched_trigger="unknown"):
        self.class0 = class0
        self.action = action
        self.exec_id_type = exec_id_type
        self.exec_id_val = exec_id_val
        self.text = text
        self.text_input = text_input
        self.isEnd = isEnd
        self.crop_screenshot_path = crop_screenshot_path
        self.state_screenshot_path = state_screenshot_path
        self.matched_trigger = matched_trigger

    def print_event(self):
        print("---- printing dest event ----")
        print("action:")
        print(self.action)
        print("exec_id_type")
        print(self.exec_id_type)
        print("exec_id_val")
        print(self.exec_id_val)
        print("text_input")
        print(self.text_input)


class TestGenerator:
    def __init__(self, desired_capabilities, app_name, out_root, text_sim_flag=True, eval_flag=False):
        self.explorer = explorer.Explorer(desired_capabilities)
        self.test_num = 0
        self.MAX_TEST_NUM = 2
        self.MAX_ACTION = 20
        self.hub_number = 2
        self.eval_flag = eval_flag
        self.bert = SentenceTransformer('bert-base-nli-mean-tokens').to('cpu')
        nltk.download('words')
        nltk.download('punkt')
        self.words = set(nltk.corpus.words.words())

        self.usage_name = usage_name
        # result output path
        output_path = os.path.join(out_root, 'output', 'models', usage_name, 'dynamic_output')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        self.output_dir = os.path.join(output_path, app_name)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        # generated test save path
        self.generated_tests_dir = os.path.join(self.output_dir, 'generated_tests')
        if not os.path.isdir(self.generated_tests_dir):
            os.makedirs(self.generated_tests_dir)
        # classified widgets save path
        self.widget_classification_res_dir = os.path.join(self.output_dir, 'widget_classifier')
        if not os.path.isdir(self.widget_classification_res_dir):
            os.makedirs(self.widget_classification_res_dir)

        # load model
        usage_model_path = os.path.join(out_root, 'output', 'models', usage_name,
                                        'usage_model-' + app_name + '.pickle')
        pickle_filepath = os.path.join(usage_model_path)
        self.usage_model = pickle.load(open(pickle_filepath, 'rb'))
        self.usage_model.states = list(set(self.usage_model.states))
        self.MG = self.usage_model.get_graph()
        self.MG.draw(os.path.join(self.output_dir, usage_name + '.png'), prog='dot')
        # build graph
        self.DG = nx.DiGraph()
        self.DG.add_edges_from(self.MG.edges())
        self.DG1 = nx.DiGraph()
        self.DG1.add_edges_from(self.MG.edges())
        self.DG1.remove_nodes_from(['start', 'end'])
        # Directed graph DG1 without start and end nodes

        if text_sim_flag:
            self.text_sim_bert = SimilarityCalculator_BERT()
        else:
            self.text_sim_bert = None

        if eval_flag:
            self.eval_results = {}
        self.tests_hubs = []
        self.hubstate_visit = {}
        self.eval_results = {}

    def is_test_equal(self, test1, test2):
        if not len(test1) == len(test2):
            return False
        i = 0
        while i < len(test1):
            if not self.is_event_equal(test1[i], test2[i]):
                return False
            i += 1
        return True

    def save_test(self, current_generated_test):
        print('saving test', self.test_num, '...')
        for test_file in glob.glob(os.path.join(self.generated_tests_dir, 'test_executable*')):
            existing_test = pickle.load(open(test_file, 'rb'))
            if self.is_test_equal(existing_test, current_generated_test):
                print('test already generated')
                return
        file_path = os.path.join(self.generated_tests_dir, 'test_executable' + str(self.test_num) + '.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(current_generated_test, file)
        print('test generated and saved!')

    def generate_test(self):
        # obtain hub nodes by centrality analysis
        hub_nodes = self.explorer.find_hub_node(self.DG1)[:self.hub_number]
        while self.test_num < self.MAX_TEST_NUM:
            self.hub_done = False
            self.step_index = 0
            self.explorer.screenshot_idx = 0
            self.pre_value_1 = []
            self.recorded_states_and_triggers = {}
            self.prescIRs = []
            # setting end flag
            is_end = False
            end_nodes = list(self.DG.predecessors('end'))
            match_scores = {}
            current_generated_test = []

            while self.step_index < self.MAX_ACTION and not is_end:
                time.sleep(5)
                if self.step_index != 0:
                    self.prescIR = self.recorded_states_and_triggers["screen"]
                    self.pretriggers = self.recorded_states_and_triggers["triggers"]
                    self.prescIRs.append(self.prescIR)
                else:
                    self.prescIR = 'start'
                    self.prescIRs.append(self.prescIR)
                    self.pretrigger = 'initial'

                scroll = input('Do you want to scroll down to see more widgets first?')
                current_state = self.explorer.extract_state(self.output_dir, self.test_num)
                self.nodes = current_state.nodes
                matching_screenIR = self.find_mathing_state_in_usage_model(current_state, end_nodes)

                self.scroll = scroll
                if scroll == 'up' or scroll == 'down' or scroll == 'left' or scroll == 'right':

                    next_event = DestEvent(class0=None, action='swipe-' + scroll,
                                           exec_id_type=None,
                                           exec_id_val=None, text='', text_input='', isEnd=False,
                                           crop_screenshot_path=None,
                                           state_screenshot_path=current_state.screenshot_path)

                    XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace(
                        '.xml', '')
                    if XML_basename in self.eval_results.keys():
                        self.eval_results[XML_basename]['true_screen_IR'] = matching_screenIR
                        self.eval_results[XML_basename]['true_widget_IR'] = scroll
                    else:
                        self.eval_results[XML_basename] = {}
                        self.eval_results[XML_basename]['true_screen_IR'] = matching_screenIR
                        self.eval_results[XML_basename]['true_widget_IR'] = scroll

                    self.recorded_states_and_triggers["screen"] = matching_screenIR
                    self.recorded_states_and_triggers["triggers"] = scroll
                    current_generated_test.append(next_event)

                    self.explorer.execute_event(next_event)
                    self.pretrigger = scroll
                    self.step_index += 1
                    continue

                else:
                    next_event_list = self.find_next_event_list(current_state, matching_screenIR, hub_nodes,
                                                                match_scores)
                    if next_event_list == self.pretrigger:
                        self.explorer.execute_event(current_generated_test[-1], current_generated_test[-1].class0)
                        current_generated_test.append(current_generated_test[-1])

                        last_key = list(self.eval_results.keys())[-1]
                        XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace(
                            '.xml', '')
                        self.eval_results[XML_basename] = self.eval_results[last_key]
                        is_end = True

                        if is_end:
                            ans = input(
                                "The model detected that the program should end now, do you want to keep going?")
                            if ans == 'y':
                                is_end = False
                        continue

                    print("List of possible events found")
                    if (self.recorded_states_and_triggers["screen"] == "sign_in") or (
                            self.recorded_states_and_triggers["screen"] == "sign_up"):
                        next_event_list = self.explorer.generate_user_input(next_event_list, current_state, match_scores)

                if next_event_list is None or next_event_list == [[]]:
                    element_candidates = []
                    for element in current_state.nodes:
                        if element.interactable:
                            image = PIL.Image.open(element.path_to_screenshot)
                            image.show()
                            element_candidates.append(element)
                    event_index = int(input(
                        'no next event found based on the usage model, please provide the index of the event to trigger (enter any out of range index to end current test)\n'))
                    # kill all the images opened by Preview
                    for proc in psutil.process_iter():
                        # print(proc.name())
                        if proc.name() == 'Preview':
                            proc.kill()
                    if event_index >= len(element_candidates):
                        break
                    else:
                        element = element_candidates[event_index]

                        widgetIR = input(
                            'Please enter the ground truth IR for the widget the input was typed on:')
                        XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace('.xml',
                                                                                                            '')
                        if XML_basename in self.eval_results.keys():
                            self.eval_results[XML_basename]['true_widget_IR'] = widgetIR
                        else:
                            self.eval_results[XML_basename] = {}
                            self.eval_results[XML_basename]['true_widget_IR'] = widgetIR
                        guided_event = DestEvent(class0=element.attributes['class'], action='click',
                                                 exec_id_type=element.get_exec_id_type(),
                                                 exec_id_val=element.get_exec_id_val(), text=element.get_text(),
                                                 text_input='', isEnd=False,
                                                 crop_screenshot_path=element.path_to_screenshot,
                                                 state_screenshot_path=current_state.screenshot_path,
                                                 matched_trigger=widgetIR)

                        current_generated_test.append(guided_event)

                        self.explorer.execute_event(guided_event)
                        self.pretrigger = guided_event.matched_trigger
                        ans = input(
                            "The model detected that the program should end now, do you want to keep going?")
                        if ans == 'n':
                            is_end = True

                elif len(next_event_list) == 1:
                    if type(next_event_list[0]) is list:
                        print('the only event is a list of the following events. should be self actions')
                        for event in next_event_list[0]:
                            print(event.exec_id_val, event.exec_id_val, event.action)
                            adopt = input('Do you want to trigger this event? y or n?')
                            if adopt == 'y':
                                current_generated_test.append(event)

                                self.explorer.execute_event(event)
                                self.pretrigger = event.matched_trigger
                                correct_widgetIR = input('Please enter the ground truth IR for the widget you chose:')
                                XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace(
                                    '.xml',
                                    '')
                                if XML_basename in self.eval_results.keys():
                                    self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                                else:
                                    self.eval_results[XML_basename] = {}
                                    self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                                for proc in psutil.process_iter():
                                    if proc.name() == 'Preview':
                                        proc.kill()
                    else:
                        image = PIL.Image.open(next_event_list[0].crop_screenshot_path)
                        image.show()
                        correct_widgetIR = input('Please enter the ground truth IR for the widget you chose:')
                        XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace('.xml', '')
                        # print("XML_basename",XML_basename)
                        if XML_basename in self.eval_results.keys():
                            self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                        else:
                            self.eval_results[XML_basename] = {}
                            self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                        for proc in psutil.process_iter():
                            if proc.name() == 'Preview':
                                proc.kill()
                        is_end = (next_event_list[0]).isEnd
                        if is_end:
                            ans = input(
                                "The model detected that the program should end now, do you want to keep going?")
                            if ans == 'y':
                                is_end = False
                        current_generated_test.append(next_event_list[0])

                        step_classification_res_dir_path = os.path.join(self.widget_classification_res_dir,
                                                                        XML_basename)
                        if not os.path.isdir(step_classification_res_dir_path):
                            os.makedirs(step_classification_res_dir_path)
                        image_name = str(0) + ".png"
                        image.save(os.path.join(step_classification_res_dir_path, image_name))
                        with open(os.path.join(step_classification_res_dir_path, "recoded_state_triggers.json"),
                                  "w") as triggers_file:
                            json.dump(self.recorded_states_and_triggers, triggers_file)

                        self.explorer.execute_event(next_event_list[0])
                        self.pretrigger = next_event_list[0].matched_trigger
                else:
                    added_in_step = 1
                    if [] in next_event_list:
                        next_event_list.remove([])
                    inscore = []

                    next_event_list1 = next_event_list.copy()

                    for next_event in next_event_list1:
                        if type(next_event) is list:  # trigger self actions first
                            for self_action in next_event:
                                if self_action.action == "send_keys":
                                    self.explorer.execute_event(self_action)
                                    self.pretrigger = self_action.matched_trigger
                                    correct_widgetIR = input(
                                        'Please enter the ground truth IR for the widget the input was typed on:')
                                    XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace(
                                        '.xml', '') + "-" + str(added_in_step)
                                    if XML_basename in self.eval_results.keys():
                                        self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                                    else:
                                        self.eval_results[XML_basename] = {}
                                        self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                                    match_scores[self_action.exec_id_val] = 0.9
                                    added_in_step += 1
                                else:
                                    next_event_list.append(self_action)

                                next_screenir = [self.recorded_states_and_triggers["screen"]]
                                score = self.explorer.intimacy_new(next_screenir, end_nodes, hub_nodes, self.DG, self.prescIRs, self.hub_done)
                                if score:
                                    inscore.append((self_action, score))

                            next_event_list.remove(next_event)
                        else:
                            if len(next_event_list) == 1:
                                inscore.append((next_event, 1))
                            else:
                                next_screenir = [j.dest for j in
                                                 self.usage_model.machine.get_transitions(
                                                     trigger=next_event.matched_trigger,
                                                     source=self.recorded_states_and_triggers["screen"])]

                                if next_screenir:
                                    score = self.explorer.intimacy_new(next_screenir, end_nodes, hub_nodes, self.DG, self.prescIRs, self.hub_done)
                                else:
                                    score = 0.01

                                if score:
                                    inscore.append((next_event, score))

                    if self.recorded_states_and_triggers["screen"] not in end_nodes:
                        inscore.sort(key=lambda x: x[1], reverse=True)
                        next_event_list0 = []
                        for i in range(len(inscore)):
                            next_event_list0.append(inscore[i][0])
                        next_event_list = next_event_list0

                    XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace('.xml', '')
                    step_classification_res_dir_path = os.path.join(self.widget_classification_res_dir, XML_basename)
                    if len(next_event_list) != 0:
                        if not os.path.isdir(step_classification_res_dir_path):
                            os.makedirs(step_classification_res_dir_path)

                    n0 = 0
                    if len(next_event_list) != 1:
                        similars = []
                        next_event_list1 = []
                        use_stopwords = True

                        for i, event in enumerate(next_event_list):
                            if event.exec_id_type == 'resource-id':
                                res = event.exec_id_val.split('/')[-1]
                                res0 = StrUtil.tokenize(event.exec_id_type, res, use_stopwords=use_stopwords)
                            else:
                                if event.exec_id_type == 'xPath':
                                    if event.text != '':
                                        res0 = event.text.split()
                                    else:
                                        res0 = []
                                else:
                                    res = event.exec_id_val
                                    res0 = StrUtil.tokenize(event.exec_id_type, res, use_stopwords=use_stopwords)

                            sim = match_scores[event.exec_id_val]
                            if self.recorded_states_and_triggers['screen'] in end_nodes:
                                usage = [self.usage_name.split('-')[-1].lower()]
                                usage = StrUtil.split_text(usage)
                                simusage = self.text_sim_bert.calc_similarity(' '.join(res0), ' '.join(usage))
                                sim = sim + simusage
                            if sim and sim > 0:
                                similars.append((event, sim))
                        similars.sort(key=lambda x: x[1], reverse=True)

                        for i in range(len(similars)):
                            next_event_list1.append(similars[i][0])

                        for i in next_event_list1:
                            if i.matched_trigger in ['item_i#click', 'category_i#click']:
                                n0 += 1
                        next_event_list = next_event_list1

                    new_next_event_list = next_event_list.copy()
                    if self.test_num > 0:
                        index_list = []
                        for index, (step, state) in enumerate(self.hubstate_visit.items()):
                            if self.step_index == int(step.split('-')[1]) and matching_screenIR == state[
                                'true_screen_IR']:
                                if 'true_widget_IR' not in state.keys():
                                    continue
                                for i, event in enumerate(next_event_list):
                                    if event.matched_trigger.split('#')[0] == state['true_widget_IR']:
                                        index_list.append(i)

                        for index_i in sorted(index_list, reverse=True):
                            if new_next_event_list[index_i].action == 'click':
                                del new_next_event_list[index_i]
                            else:
                                index_list.remove(index_i)
                        for index_i in index_list:
                            new_next_event_list.append(next_event_list[index_i])

                    next_event_list = self.explorer.adjust_events(new_next_event_list, 'exec_id_val')
                    next_event_list = self.explorer.adjust_events(next_event_list, 'matched_trigger')

                    for i, event in enumerate(next_event_list[:5 + n0]):
                        print(event.exec_id_val)
                        print("id:" + str(
                            i) + " - val: " + event.exec_id_val + "- matched with: " + event.matched_trigger)
                        image = PIL.Image.open(event.crop_screenshot_path)
                        image.show()
                        image_name = str(i) + "-" + event.matched_trigger + ".png"
                        image.save(os.path.join(step_classification_res_dir_path, image_name))

                    with open(os.path.join(step_classification_res_dir_path, "recoded_state_triggers.json"),
                              "w") as triggers_file:
                        json.dump(self.recorded_states_and_triggers, triggers_file)
                    event_indx = input('Choose the id of the widget you want to interact with:')
                    next_event = next_event_list[int(event_indx)]

                    if next_event.action == "send_keys":
                        input_text = input('Type in the input for the chosen widget:')
                        next_event.text_input = input_text
                    correct_widgetIR = input('Please enter the ground truth IR for the widget you chose:')

                    if added_in_step != 1:
                        XML_basename += "-" + str(added_in_step)
                    if XML_basename in self.eval_results.keys():
                        self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR
                    else:
                        self.eval_results[XML_basename] = {}
                        self.eval_results[XML_basename]['true_widget_IR'] = correct_widgetIR

                    next_event.matched_trigger = next_event.matched_trigger.replace(
                        next_event.matched_trigger.split("#")[0], correct_widgetIR)

                    correct_widgetIRlist = StrUtil.tokenize('IR', correct_widgetIR, True)
                    correct_widgetIR0 = " ".join(correct_widgetIRlist)
                    if not self.hub_done and self.step_index != 0 and matching_screenIR not in self.tests_hubs:
                        self.hub_done = self.is_hub_done(correct_widgetIR0, hub_nodes)

                        if not self.hub_done:
                            if matching_screenIR in hub_nodes:
                                self.hub_done = True

                        if self.hub_done:
                            visited = '-'.join([str(self.test_num), str(self.step_index)])
                            if visited not in self.hubstate_visit.keys():
                                self.hubstate_visit[visited] = {}
                                self.hubstate_visit[visited] = self.eval_results[visited]
                                print(self.hubstate_visit)

                    is_end = next_event.isEnd

                    if is_end or self.prescIR == matching_screenIR:
                        ans = input("The model detected that the program should end now, do you want to keep going?")
                        if ans == 'y':
                            is_end = False
                        else:
                            is_end = True

                    class0 = None
                    for proc in psutil.process_iter():
                        if proc.name() == 'Preview':
                            proc.kill()
                    if next_event.exec_id_type == 'text':
                        for node in self.nodes:
                            if 'rotation' in node.attributes.keys() or 'text' not in node.attributes.keys():
                                continue
                            if node.attributes['text'] == next_event.exec_id_val:
                                class0 = node.attributes['class']
                                next_event.class0 = class0
                    self.explorer.execute_event(next_event, class0)
                    current_generated_test.append(next_event)
                    self.pretrigger = next_event.matched_trigger

                self.step_index += 1
            self.save_test(current_generated_test)
            self.explorer.driver.close_app()

            with open(os.path.join(self.output_dir, 'eval_results' + '.json'), 'a') as outfile:
                json.dump(self.eval_results, outfile)
            self.explorer.driver.launch_app()
            self.test_num += 1

    def is_event_equal(self, event1, event2):
        if event1.exec_id_type == event2.exec_id_type and event1.exec_id_val == event2.exec_id_val:
            return True
        return False

    def is_hub_done(self, IR, hub_nodes):
        done_flag = False
        if IR in hub_nodes:
            done_flag = True
            self.tests_hubs.append(IR)
        if not done_flag:
            for hubnode in hub_nodes:
                if set(IR).issubset(hubnode):
                    if self.prescIR not in self.tests_hubs:
                        self.tests_hubs.append(hubnode)
                        done_flag = True
                        break
        return done_flag

    def find_mathing_state_in_usage_model(self, current_state, end_nodes):
        XML_basename = os.path.basename(os.path.normpath(current_state.UIXML_path)).replace('.xml', '')
        cans_screen = self.explorer.getcandidates_screenIR(self.DG, self.pretrigger, self.step_index, self.prescIR)
        if self.step_index == 1 and self.prescIR not in ['popup', 'get_started', 'home']:
            successors_successors = []
            for s in list(self.DG.successors('start')):
                successors_successors.extend(self.DG.successors(s))
            cans_screen.extend(successors_successors)
            cans_screen = list(set(cans_screen))
        if 'end' in cans_screen:
            cans_screen.remove('end')

        if self.prescIR in end_nodes and self.prescIR not in cans_screen:
            cans_screen.append(self.prescIR)

        if self.usage_name.split('-')[-1].lower() in ['removecart'] and self.prescIR in "cart":
            cans_screen.append('confirm_remove')

        if self.step_index > 0:
            cans_screen = list(set(self.pre_cans_screen).union(set(cans_screen)))
        self.pre_cans_screen = cans_screen

        self.similardict, self.operable_widgets, self.value_1 = current_state.semantic_screen_classifiers(cans_screen, self.prescIR, self.text_sim_bert, self.usage_model, self.DG1, self.pretrigger,
            self.usage_name)

        set1 = set(tuple(d.items()) for d in self.value_1)
        set2 = set(tuple(d.items()) for d in self.pre_value_1)
        intersection = set1.intersection(set2)
        common = [dict(item) for item in intersection]

        screen_IR_candidates = sorted(self.similardict, key=lambda k: self.similardict[k][0], reverse=True)

        top5 = screen_IR_candidates[:5]
        if len(common) > 0 and abs(len(self.value_1) - len(self.pre_value_1)) < 5 or self.pretrigger in ['up', 'down',
                                                                                                         'left',
                                                                                                         'right']:
            if self.prescIR in top5:
                top5.remove(self.prescIR)
                screen_IR_candidates.remove(self.prescIR)

            top5.insert(0, self.prescIR)
            screen_IR_candidates.insert(0, self.prescIR)

        self.pre_value_1 = self.value_1

        if top5:
            print("The screen classifier top5 guesses for the screen: ", top5)

        current_screenIR = input('\nChoose the closest screen tag from the top5 guesses:\n')
        # correct_screenIR = input('\nType in the correct screen tag:\n')

        correct_screenIR = current_screenIR
        if XML_basename in self.eval_results.keys():
            self.eval_results[XML_basename]['true_screen_IR'] = correct_screenIR
        else:
            self.eval_results[XML_basename] = {}
            self.eval_results[XML_basename]['true_screen_IR'] = correct_screenIR

        triggers = self.usage_model.machine.get_triggers(current_screenIR)

        if len(triggers) == 0:
            raise ValueError('current screenIR does not have any triggers...')
        else:
            return current_screenIR

    def is_widgetIR_input_type(self, widgetIR):
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        widgetIR_file = os.path.join(current_dir_path, '..', '..', 'IR', 'widget_ir.csv')
        df = pd.read_csv(widgetIR_file)
        row_found = df.loc[df['ir'] == widgetIR]
        if len(row_found) == 0:
            print('no widget IR found in the IR definition, check IR', widgetIR)

        elif row_found.iloc[0]['widget_type'] == 'input':
            return True
        return False

    def find_actions_from_self_transition(self, matching_screenIR, current_state, match_scores):

        self_actions = []
        all_triggers = self.usage_model.machine.get_triggers(matching_screenIR)
        self_actions_added = set()
        needs_user_input = False

        for condition in self.condition_list:
            print('finding element for------', condition)
            if '#' in condition:
                widgetIR = condition.split('#')[0]
                action = condition.split('#')[1]

                if self.is_widgetIR_input_type(widgetIR):
                    needs_user_input = True
                elif condition in all_triggers:
                    pass  # if current self action is covered by other triggers, it means this self action can jump to a diff state, so skip it and handle it when it appears in other triggers (that's not self trigger)
                else:
                    top_candidates, heuristic_matches, matchsimilar = current_state.find_widget_candidates(
                        widgetIR, matching_screenIR, self.operable_widgets, self.similardict, self.text_sim_bert,
                        self.usage_name)

                    for element in heuristic_matches + top_candidates:
                        if element.get_exec_id_val() not in self_actions_added:
                            self.append_action(self_actions, element, action, '', current_state, condition)
                            self_actions_added.add(element.get_exec_id_val())
                            match_scores[element.get_exec_id_val()] = matchsimilar[element]

        # generate actions for EditText fields and fill the form
        if needs_user_input:
            print('generating user inputs...')
            for element in current_state.nodes:
                if element.interactable and 'EditText' in element.get_element_type():
                    image = PIL.Image.open(element.path_to_screenshot)
                    image.show()
                    print("element.get_exec_id_val()", element.get_exec_id_val())
                    user_input = input(
                        'please enter your input for element that was just opened\n enter nothing if you want to skip this element\n')
                    if not user_input == '':
                        self.append_action(self_actions, element, 'send_keys', user_input, current_state, condition)

                    for proc in psutil.process_iter():
                        if proc.name() == 'Preview':
                            proc.kill()

        return self_actions

    def find_next_event_list(self, current_state, matching_screenIR, hub_nodes, match_scores):
        next_event_list = []
        if matching_screenIR is None:
            print('no matching state found in the usage model...')
            return []
        else:
            if not self.hub_done and self.step_index > 1:
                if matching_screenIR in hub_nodes and matching_screenIR not in self.tests_hubs:
                    self.hub_done = True
                    self.tests_hubs.append(matching_screenIR)
                    visited = '-'.join([str(self.test_num), str(self.step_index - 1)])
                    if visited not in self.hubstate_visit.keys():
                        self.hubstate_visit[visited] = {}
                        self.hubstate_visit[visited] = self.eval_results[visited]

            self.pre_states_and_triggers = self.recorded_states_and_triggers
            all_possible_triggers = self.usage_model.machine.get_triggers(matching_screenIR)
            if matching_screenIR == 'get_started' and 'continue#click' not in all_possible_triggers:
                all_possible_triggers.insert(0, 'continue#click')
            if matching_screenIR == 'menu':
                if 'menu_settings#click' not in all_possible_triggers:
                    all_possible_triggers.insert(0, 'menu_settings#click')

                if self.usage_name.split('-')[-1].lower() in ['help', 'contact']:
                    if 'help' not in all_possible_triggers or 'contact' not in all_possible_triggers:
                        all_possible_triggers.insert(0, 'contact#click')
                        all_possible_triggers.insert(0, 'help#click')
            if matching_screenIR == 'search':
                if 'search#click' not in all_possible_triggers:
                    all_possible_triggers.insert(0, 'search#click')

            if matching_screenIR == 'home' and 'menu#click' not in all_possible_triggers:
                all_possible_triggers.insert(0, 'menu#click')

            # all_possible_triggers ['menu#click', 'continue#click', 'self', 'to_signin_or_signup#click']
            self.recorded_states_and_triggers = {
                "screen": matching_screenIR,
                "triggers": all_possible_triggers
            }

            if 'self' in all_possible_triggers:
                self.condition_list = []
                self_transitions = self.usage_model.machine.get_transitions(trigger='self',
                                                                            source=matching_screenIR,
                                                                            dest=matching_screenIR)
                condition_list = self.usage_model.get_condition_list(self_transitions)
                for condition in condition_list:
                    if condition in ['up', 'down', 'left', 'right'] and condition != self.scroll:
                        continue

                    self.condition_list.append(condition)
                if self.condition_list:
                    self_actions = self.find_actions_from_self_transition(matching_screenIR, current_state,
                                                                          match_scores)
                    next_event_list.append(self_actions)
                all_possible_triggers.remove('self')

            possible_actions = self.explorer.find_possible_next_actions_new(current_state, matching_screenIR,
                                                                   all_possible_triggers, match_scores, self.usage_model, self.operable_widgets, self.similardict, self.text_sim_bert, self.usage_name, self.nodes)

            for possible_action in possible_actions:
                next_event_list.append(possible_action)

        return next_event_list
        # In the end your next event is a combination of a widget (next_event_widget) which is the type of the
        # node object defined in node.py. and and action that can be either "click" or "send_keys" or
        # "send_keys_enter" or "long" or "swipe-up" etc. You can use the code line below to make a DestEvent (which is defined at the top of this file) - if your action type is send keys then the text input
        # argument would be the input ow it would be empty string

    def append_action(self, self_actions, element, action, user_input, current_state, condition):
        self_actions.append(DestEvent(
            class0=element.attributes['class'],
            action=action,
            exec_id_type=element.get_exec_id_type(),
            exec_id_val=element.get_exec_id_val(),
            text=element.get_text(),
            text_input=user_input,
            isEnd=False,
            crop_screenshot_path=element.path_to_screenshot,
            state_screenshot_path=current_state.screenshot_path,
            matched_trigger=condition
        ))


if __name__ == "__main__":

    AUT = Wish()
    usage_name = '1-SignIn'
    final_data_root = FINAL_ARTIFACT_ROOT_DIR
    test_gen = TestGenerator(AUT.desiredCapabilities, AUT.appname, final_data_root)
    start = time.time()
    test_gen.generate_test()
    end = time.time()
    print("Dynamic generation running time " + str(end - start) + " seconds")
    # kill all the images opened by Preview
    for proc in psutil.process_iter():
        # print(proc.name())
        if proc.name() == 'Preview':
            proc.kill()
