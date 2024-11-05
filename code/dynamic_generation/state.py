import sys, os
import csv
import PIL, psutil
import pandas as pd
import cv2
import json
import pytesseract
import numpy as np

current_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir_path, 'similarity_calculate'))


from StrUtil import StrUtil


def OCR(path, words_only=True):
    image = cv2.imread(path)
    image = cv2.medianBlur(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(PIL.Image.fromarray(image))
    text = text.split()
    return [word.lower() for word in text]


def process_text_data(raw_data, words):
    remove_none_english = [w for w in raw_data if w in words]
    sentence = " ".join(remove_none_english)
    return sentence


class State:
    def __init__(self, screenshot):
        self.screenshot = screenshot
        # print("self.screenshot",self.screenshot)
        # self.screenshot.show()
        self.nodes = []
        self.actions = {}
        self.name_actions = {}
        self.activity = ''
        self.transitions = {}
        self.screenshot_path = ''
        self.UIXML_path = ''

    def add_screenshot_path(self, path):
        self.screenshot_path = path

    def add_UIXML_path(self, path):
        self.UIXML_path = path

    def add_node(self, node):
        self.nodes.append(node)

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_actions(self):
        return self.actions

    def get_name_actions(self):
        return self.name_actions

    def add_action(self, node_id, tag, action_type):
        self.actions[node_id] = action_type
        self.name_actions[tag] = action_type

    def set_activity(self, activity_name):
        self.activity = activity_name

    def add_transition(self, action, state):
        self.transitions[action] = state

    def print_state(self):
        print('Activity:', self.activity)
        print('Actions:', self.name_actions)
        print("-------------------")
        for node in self.nodes:
            if node.interactable:
                print(node.get_exec_identifiers())
        print("-------------------")

    def get_screen_tag(self, screen_id):
        screen_dict_path = "../4_dynamic_generation/screen_classifier/screen_dict.json"
        with open(screen_dict_path) as screen_dict_file:
            screen_dict = json.load(screen_dict_file)
            for k, v in screen_dict.items():
                if v == screen_id:
                    return k

    def get_text_list(self, node, widget):
        sent_semantic_text = []
        word_semantic_text = []
        if 'text' in widget.keys() and widget['text'] == '':
            node.add_ocr_to_data()
            if node.data["ocr"] != '':
                if len(node.data["ocr"].split()) < 7:
                    res1 = StrUtil.tokenize("ocr", node.data["ocr"], use_stopwords=True)
                    sent_semantic_text.extend(res1)

                    res0 = StrUtil.tokenize("ocr", node.data["ocr"], use_stopwords=True)
                    word_semantic_text.append(res0)

        for key, value in widget.items():
            if key == 'class':
                continue
            if key == 'content-desc' and 'text' in widget.keys() and widget['text'] != '':
                continue

            res0 = StrUtil.tokenize(key, widget[key], use_stopwords=True)
            res = [item.lower() for item in res0]

            if key == "text":
                if len(res) > 10:
                    continue

            sent_semantic_text.extend(res)
            if res not in word_semantic_text:
                word_semantic_text.append(res)

        sent_semantic_text = list(dict.fromkeys(sent_semantic_text))
        all_words = [word for phrase in sent_semantic_text for word in phrase.split()]
        # 从列表尾部开始检查重复元素，并删除索引大的重复元素
        i = len(all_words) - 1
        while i > 0:
            if all_words[i] in all_words[:i]:
                all_words.pop(i)
            i -= 1
        sent_semantic_text = all_words
        return sent_semantic_text, word_semantic_text

    def get_screenIRwordlist(self, screenIR):
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        wordlist_dir = os.path.join(current_dir_path, '..', '..', 'IR')
        wordlist_path = os.path.join(wordlist_dir, 'screen_ir.csv')

        if os.path.exists(wordlist_path):
            with open(wordlist_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['ir'] == screenIR:
                        description = row['description']
                        description = description.replace("(", "").replace(")", "")
                        return description
            return screenIR
        else:
            return screenIR  # if no wordlist is found for the screenIR, use the screenIR name itself as the wordlist

    def get_widgetwordlist(self, widgetIR):
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        wordlist_dir = os.path.join(current_dir_path, '..', '..', 'IR')
        wordlist_path = os.path.join(wordlist_dir, 'widget_ir.csv')

        if os.path.exists(wordlist_path):
            with open(wordlist_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['ir'] == widgetIR:
                        description = row['description']

                        description = description.replace("(", "").replace(")", "")
                        return description
                    else:
                        wordlist1 = StrUtil.tokenize('IR', widgetIR, True)
                        return ' '.join(wordlist1)
        else:
            wordlist1 = StrUtil.tokenize('IR', widgetIR, True)
            return ' '.join(wordlist1)

    def find_widget_to_trigger(self, widgetIR):
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        widget_ir_csv = os.path.join(current_dir_path, '..', '..', 'IR', 'widget_ir.csv')
        widget_df = pd.read_csv(widget_ir_csv)
        row_found = widget_df.loc[widget_df['ir'] == widgetIR]

        if len(row_found) == 0:
            raise ValueError('no widget IR found')
        else:
            widget_type = row_found['widget_type'].values[0]

        input_element_type = ['EditText', 'AutoCompleteTextView', 'Spinner']
        element_candidates = []

        for element in self.nodes:

            if element.interactable:
                element_type = element.get_element_type().split('.')[-1]
                if (widget_type == 'input' and element_type not in input_element_type) \
                        or (pd.isna(widget_type) and element_type in input_element_type):
                    continue
                element_candidates.append(element)
        for element in element_candidates:
            image = PIL.Image.open(element.path_to_screenshot)
            image.show()
            # element.data # has content-desc, resource-id, text
        i = int(input('widget index to trigger\n'))
        # kill all the images opened by Preview
        for proc in psutil.process_iter():
            # print(proc.name())
            if proc.name() == 'Preview':
                proc.kill()
        if i >= len(element_candidates):
            return None
        return element_candidates[i]

    def find_widget_candidates(self, widgetIR, screenIR, operable_widgets, similardict, text_sim_bert, usage_name):

        top_candidates = []
        heuristic_matches = []
        screen_value = similardict[screenIR]
        pos0 = screen_value[3]
        matchsimilar = {}
        max_v = screen_value[2][0]
        threshold = 0.65
        if max_v < threshold:
            threshold = 0.45

        for i, (node, ir) in enumerate(pos0):
            if ir == widgetIR and screen_value[2][i] > threshold:
                if widgetIR in ['category_i', 'item_i'] and type(node) == list:
                    top_candidates.extend(node)
                    for node_i in node:
                        matchsimilar[node_i] = screen_value[2][i]
                else:
                    top_candidates.append(node)
                    matchsimilar[node] = screen_value[2][i]

        # for element in operable_widgets.keys():
        # for index, (element, value) in enumerate(operable_widgets.items()):
        #     # print(index, key, value)
        #     senmatic_text, wordtext_info = self.get_text_list(element, value)
        #     text=senmatic_text
        #
        #     if text == []:
        #         continue
        #
        #     # text = element.get_processed_textual_info()
        #     # text为ocr识别的文本，以及元素属性处理过后的合并连接一起的文本
        #
        #     element_type = element.get_element_type().split('.')[-1]
        #     if element_type == "ScrollView" or element_type == "RecyclerView":
        #         continue
        #     # 判定，如果图像和textview一起，且图像和文字是不可点击的，但是viewGroup是可以点击的，则分为一组。不能考虑在外。
        #
        #     if element_type == "ViewGroup":
        #         if element.attributes['clickable']!="true":
        #             continue
        #
        #     if (("email" in text) and element_type == "EditText") or (("password" in text) and element_type == "EditText"):
        #         continue
        #
        #
        #     # if self.look_for_exact_match(widgetIR, text,usage_name):
        #     #     if widgetIR == "search_bar":
        #     #         if element_type == "EditText":
        #     #             heuristic_matches.append(element)
        #     #             matchsimilar[element] = 0.7
        #     #             continue
        #     #     elif widgetIR == "to_search":
        #     #         if element_type != "EditText":
        #     #             heuristic_matches.append(element)
        #     #             matchsimilar[element] = 0.7
        #     #             continue
        #     #     else:
        #     #         heuristic_matches.append(element)
        #     #         matchsimilar[element] = 0.7
        #     # print("heuristic_matches",heuristic_matches)
        #     # if self.check_for_top_match_heuristics(widgetIR, screenIR, element):
        #     #     if element not in top_candidates:
        #     #         top_candidates.append(element)
        #     #         print("111111")
        #     #         matchsimilar[element] = 0.7
        #     # print("top_candidates",top_candidates)
        #     # 以上为启发式规则保留

        return top_candidates, heuristic_matches, matchsimilar

    def find_similarpair(self, similarity_matrix1, similar_values, similarity_matrix, pos0, poscan, widgets,
                         total_widgetsIR):

        threshold = 0.65
        similar_values0 = []
        if np.max(similarity_matrix1) < 0.65:
            threshold = 0.45
        while similarity_matrix1.size > 0:
            max_value = np.max(similarity_matrix1)
            if max_value >= threshold:
                similar_values0.append(max_value)
            else:
                break
            max_index = np.unravel_index(similarity_matrix1.argmax(), similarity_matrix1.shape)
            pair = (max_index[0], max_index[1])
            similarity_matrix1 = np.delete(similarity_matrix1, pair[0], axis=0)
            similarity_matrix1 = np.delete(similarity_matrix1, pair[1], axis=1)

        max_positions = []
        for max_value in similar_values0:
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    if similarity_matrix[i][j] == max_value:
                        position = (i, j)
                        if position not in max_positions:
                            is_duplicate = False
                            for item in max_positions:
                                if item[0] == position[0]:
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                max_positions.append(position)
                                pos0.append((list(widgets.items())[i][0], total_widgetsIR[j]))
                                similar_values.append(similarity_matrix[i][j])
                                poscan.append((list(widgets.items())[i][1], total_widgetsIR[j]))
                        else:
                            continue
        return similarity_matrix1, pos0, poscan, similar_values

    def analyze_matrix(self, similarity_matrix1, widgets, total_widgetsIR, widget_self, zones, recynodes, screen):

        com_flag = False
        pos0 = []
        poscan = []
        similar_values = []
        # Iterate until the similarity matrix is empty

        if len(zones.keys()) == 1 and recynodes == []:
            similarity_matrix1, pos0, poscan, similar_values = self.find_similarpair(similarity_matrix1,
                                                                                     similar_values,
                                                                                     similarity_matrix1, pos0,
                                                                                     poscan, widgets,
                                                                                     total_widgetsIR)
        else:
            if recynodes:
                node_text = []
                for ele in recynodes:
                    if 'text' not in ele.exec_identifier.keys() or ele.exec_identifier['text'] == '':
                        if 'content-desc' in ele.attributes.keys() and ele.attributes['content-desc'] != '':
                            node_text.extend(ele.attributes['content-desc'])
                        elif 'ocr' in ele.data.keys():
                            node_text.extend(StrUtil.tokenize("ocr", ele.data["ocr"]))
                        elif 'resource-id' in ele.exec_identifier.keys() and ele.exec_identifier['resource-id'] != '':
                            node_text.append(ele.attributes['resource-id'].split("/")[-1])
                    else:
                        node_text.extend(ele.exec_identifier['text'].split())

                if len(node_text) > 5 and len(recynodes) < 6 or len(node_text) < 40 and len(recynodes) >= 4 or len(
                        node_text) < 20 and len(recynodes) == 1:
                    com_flag = True

            if com_flag:
                col = []
                if len(node_text) < 40 and len(recynodes) >= 4 and 'category_i' in total_widgetsIR:
                    pos0.append((recynodes, 'category_i'))
                    for i in range(len(recynodes)):
                        similar_values.append(1.0)
                    col.append(total_widgetsIR.index('category_i'))
                else:
                    if 'item_i' in total_widgetsIR:
                        pos0.append((recynodes, 'item_i'))
                        for i in range(len(recynodes)):
                            similar_values.append(1.0)
                        col.append(total_widgetsIR.index('item_i'))

                recy_rows = []
                for node in recynodes:
                    index = list(widgets.keys()).index(node)
                    recy_rows.append(index)

                if recy_rows != [] and col != []:
                    for row_index in recy_rows:
                        for i in range(len(similarity_matrix1[row_index])):
                            if similarity_matrix1[row_index][i] != 0.9:
                                similarity_matrix1[row_index][i] = 0

                    for column_index in col:
                        for i in range(len(similarity_matrix1)):
                            if similarity_matrix1[i][column_index] != 0.9:
                                similarity_matrix1[i][column_index] = 0

            similarity_matrix1, pos0, poscan, similar_values = self.find_similarpair(similarity_matrix1,
                                                                                     similar_values,
                                                                                     similarity_matrix1, pos0,
                                                                                     poscan, widgets,
                                                                                     total_widgetsIR)

        return similar_values, pos0

    def get_word_simval(self, widgetlist, IRlist, text_sim_bert):

        max_similarity = 0  # 初始化最高相似度为-1
        if [] in widgetlist:
            widgetlist.remove([])

        for words in widgetlist:
            if words == IRlist:
                similarity = 1
            else:
                similarity = text_sim_bert.calc_similarity(' '.join(words), ' '.join(IRlist))[0][0]

            max_similarity = max(max_similarity, similarity)
        return max_similarity

    def convert_bounds_to_screen_zone(self, bounds):
        zones = {1: [0, 315, 1080, 1636], 2: [0, 0, 1080, 315], 3: [0, 1636, 1080, 1794]}
        # Set 1 to the middle position, 2 to the upper navigation bar position, and 3 to the lower navigation bar position

        x = bounds[0]
        y = bounds[1]
        for zone, zone_bounds in zones.items():
            if zone_bounds[0] <= x <= zone_bounds[2] and zone_bounds[1] <= y <= zone_bounds[3]:
                return zone
        return 0

    def convert_bounds_to_wIR(self, bounds):
        zones = {'menu': [0, 66, 154, 220], 'search_bar': [159, 89, 447, 184]}

        x = bounds[0]
        y = bounds[1]
        for zone, zone_bounds in zones.items():
            if zone_bounds[0] <= x <= zone_bounds[2] and zone_bounds[1] <= y <= zone_bounds[3]:
                return zone
        return 0

    def is_inbounds(self, node_bounds, midpoint):

        points = node_bounds.split("][")
        boundslist = [int(points[0].split(",")[0][1:]), int(points[0].split(",")[1]), int(points[1].split(",")[0]),
                      int(points[1].split(",")[1][:-1])]

        x = midpoint[0]
        y = midpoint[1]
        if boundslist[0] <= x <= boundslist[2] and boundslist[1] <= y <= boundslist[3]:

            return True
        else:
            return False

    def semantic_screen_classifiers(self, cans_screen, prescIR, text_sim_bert, usage_model, digraph, pre_trigger,
                                    usage_name):

        triggers_total = {}
        for screeni in cans_screen:
            triggers = usage_model.machine.get_triggers(screeni)
            if prescIR == 'popup' and screeni == 'get_started':
                continue
            widgetIRs = []
            totalaction = []
            for trigger in triggers:
                widgetIR = trigger.split('#')[0]
                widgetIRs.append(widgetIR)
                action = trigger.split('#')[-1]
                if action not in totalaction:
                   totalaction.append(action)

            if 'self' in triggers:
                selfwidgetIR = []
                selftransitions = usage_model.machine.get_transitions(trigger='self', source=screeni, dest=screeni)
                condition_list = usage_model.get_condition_list(selftransitions)

                for condition in condition_list:
                    if '#' in condition:
                        widgetIR = condition.split('#')[0]
                        selfwidgetIR.append(widgetIR)
                        action = condition.split('#')[-1]

                        if action not in totalaction:
                            totalaction.append(action)

                widgetIRs.append(selfwidgetIR)
                widgetIRs.remove('self')
                totalaction.remove('self')

            if screeni == 'get_started':
                if 'continue' not in widgetIRs:
                    widgetIRs.insert(0, 'continue')

            if screeni == 'search':
                if 'search' not in widgetIRs:
                    widgetIRs.insert(0, 'search')

            if screeni == 'menu':
                if 'menu_settings' not in widgetIRs:
                    widgetIRs.insert(0, 'menu_settings')

                if usage_name.split('-')[-1].lower() in ['help', 'contact']:
                    if 'help' not in widgetIRs or 'contact' not in widgetIRs:
                        widgetIRs.insert(0, 'contact')
                        widgetIRs.insert(0, 'help')

            if screeni == 'home':
                if 'menu' not in widgetIRs:
                    widgetIRs.insert(0, 'menu')

            triggers_total[screeni] = widgetIRs

        FEATURE_KEYS = ['text', 'resource-id', 'content-desc']

        widgets = {}
        zones = {}
        Recy_flag = False
        res_value = False
        activity_wordlist = self.activity.replace('.', ' ').lower().strip()

        status = []
        recynodes = []
        recynodes0 = []
        bottom_bounds = 0
        recy_widgets = {}

        for element in self.nodes:
            if 'rotation' in element.attributes.keys():
                continue
            element_type = element.get_element_type().split('.')[-1]

            if element_type in ['LinearLayout', 'LinearLayoutCompat']:
                if element.attributes['resource-id'] != '':
                    resource_id_list = element.attributes['resource-id'].split('/')[-1].split('_')
                    if 'bottom' in resource_id_list:
                        bottom_bounds = element.attributes['bounds']

            if 'selected' in element.attributes.keys() and element.attributes['selected'] == "true":
                if element.attributes['text'] != "":
                    res0 = StrUtil.tokenize('text', element.attributes['text'], use_stopwords=True)
                    res = [item.lower() for item in res0]
                    status.append(res)



            if element_type == "RecyclerView":
                Recy_flag = True
                recynodes = element.children
                res_id_list = []
                i = len(recynodes) - 1
                remove_node = []
                while i >= 0:
                    res_id_list.append(recynodes[i].attributes['resource-id'])
                    if recynodes[i].attributes['clickable'] == 'false':
                        click_flag = False
                        for recy_child in recynodes[i].children:
                            if recy_child.attributes['clickable'] == 'true':
                                click_flag = True
                                break

                        if not click_flag:
                            remove_node.append(recynodes[i])

                    if self.convert_bounds_to_screen_zone(recynodes[i].get_middle_point()) == 3:
                        remove_node.append(recynodes[i])

                    i = i - 1

                recynodes = [x for x in recynodes if x not in remove_node]

            if element.interactable:
                d = {}
                abrog_flag = False

                if totalaction == ['click']:
                    if element.attributes['clickable'] == 'false' and element.attributes['long-clickable'] == 'false':
                        continue

                if element.num_of_children == 0:

                    for key in FEATURE_KEYS:
                        if 'image' in element.attributes[key].split() and element.get_element_type().split('.')[-1] in [
                            'ImageView']:
                            break
                        if key in element.attributes.keys() and element.attributes[key] != "":
                            d[key] = element.attributes[key]
                else:
                    if element_type in ["ViewGroup", "LinearLayout", "LinearLayoutCompat", "FrameLayout",
                                        "RelativeLayout", "Tab"]:
                        if len(element.children) != 0:
                            if len(element.children) == 1 and element.children[0].get_element_type().split('.')[-1] in [
                                "ViewGroup", "LinearLayout"] and element.children[0].attributes['clickable'] == 'true':
                                continue
                            # print("element.children",element.children)
                            if len(element.children) == 1 and element.children[0].get_element_type().split('.')[-1] in [
                                "ImageView"] and element.children[0].attributes['clickable'] == 'false':
                                for key in FEATURE_KEYS:
                                    if key in element.attributes.keys() and element.attributes[key] != "":
                                        d[key] = element.attributes[key]
                            else:
                                # if element.children[0].get_element_type().split('.')[-1] in ["ViewGroup", "LinearLayout"]
                                d, element = self.get_childrens_semantic(element, element.children)

                if d != {}:
                    if self.convert_bounds_to_screen_zone(element.get_middle_point()) == 2:
                        widgets[element] = d
                    else:
                        if Recy_flag:
                            if bottom_bounds != 0 and self.is_inbounds(bottom_bounds, element.get_middle_point()):
                                widgets[element] = d
                            else:
                                widgets[element] = d
                                is_contained = False
                                for i, recynode in enumerate(recynodes):
                                    if self.is_inbounds(recynode.attributes['bounds'], element.get_middle_point()):
                                        if i not in recy_widgets.keys():
                                            recy_widgets[i] = [(element, d)]
                                        else:
                                            recy_widgets[i].append((element, d))
                                        is_contained = True
                                if is_contained:
                                    for i, value in recy_widgets.items():
                                        max_text = 0
                                        for (recynode, d) in value:
                                            recynodes0.append(recynode)
                                            if 'text' in d.keys() and d['text'] != "":
                                                if len(d['text'].split()) > max_text:
                                                    max_text = len(d['text'].split())
                                                    widgets[recynode] = d
                                                    recynode.exec_identifier = d

                                else:
                                    widgets[element] = d
                        else:
                            widgets[element] = d
                    # print("widgets",widgets)

                    zone = self.convert_bounds_to_screen_zone(element.get_middle_point())
                    if zone not in zones.keys():
                        zones[zone] = [element]
                    else:
                        zones[zone].append(element)

        recynodes0 = list(set(recynodes0))
        for i in range(len(recynodes0) - 1, -1, -1):
            if recynodes0[i] not in widgets.keys():
                del recynodes0[i]
                continue
            points = recynodes0[i].attributes['bounds'].split("][")
            if float(points[1].split(",")[1][:-1]) - float(points[0].split(",")[1]) > 1000:
                del recynodes0[i]
                continue

        widgets, recynodes0, zones = self.process(widgets, recynodes0, bottom_bounds, zones)
        # if res_value and Recy_flag:
        #     for key in list(d.keys()):
        #         if key == 'resource-id':
        #             del d[key]
        # print("d",d)

        # widgets字典 {node1：{'resource-id': 'com.guardian:id/vpSlides'}, node2：{'resource-id': 'com.guardian:id/bUpgradeToPremium'}, {'resource-id': 'com.guardian:id/bContinue'},
        # {'resource-id': 'com.guardian:id/llAlreadySubscribed'}}
        # 得出页面中可操作的所有组件的代表语义信息
        # print("zones",zones)
        # zones {1: [<node.Node object at 0x0000015C648DC7F0>, <node.Node object at 0x0000015C648DC820>, <node.Node object at 0x0000015C648DC7C0>]}

        # triggers_total={'home': ['menu_more', 'account', 'menu', 'category_i'],'category': ['continue', 'menu_settings', 'category_i', 'to_search'],
        #  'get_started': ['item_i', 'continue', 'bypass', 'to_home'], 'popup': ['continue', 'apply', 'deny'],'items': ['item_i', 'menu_more', 'account']}
        similardict = {}

        sent_text_info = []
        word_text_info = []

        for node, current_i in widgets.items():
            senmatic_text, wordtext_info = self.get_text_list(node, current_i)
            sent_text_info.append(senmatic_text)
            word_text_info.append(wordtext_info)

            # print("node.attributes",node.attributes)

        self.sent_text_info = sent_text_info
        text_info_strs = ' '.join([item for sublist in sent_text_info for item in sublist if len(sublist) < 10])

        value_1 = []

        for widget_i, data_i in widgets.items():
            if self.convert_bounds_to_screen_zone(widget_i.get_middle_point()) == 1:
                value_1.append(data_i)

        if Recy_flag:
            extension_Activity = 'table list'
            activity_wordlist = activity_wordlist + ' ' + extension_Activity

        transitions_list = vars(usage_model.machine)['_markup']['transitions']
        for screen, widgetsIR in triggers_total.items():
            recynodes00 = recynodes0
            if screen in ['get_started', 'popup']:
                recynodes00 = []
            newwidgetsIR = []
            for item in widgetsIR:
                if type(item) == list:
                    widget_self = item
                    for subitem in item:
                        newwidgetsIR.append(subitem)
                else:
                    newwidgetsIR.append(item)
                    widget_self = []

            if screen in ['category', 'items']:
                if screen == 'category':
                    if 'category_i' not in newwidgetsIR:
                        newwidgetsIR.append('category_i')
                else:
                    if 'item_i' not in newwidgetsIR:
                        newwidgetsIR.append('item_i')

            if 'item_i' not in newwidgetsIR and 'category_i' not in newwidgetsIR:
                recynodes00 = []
            new_widgetsIR = []
            for item in newwidgetsIR:
                if item not in new_widgetsIR:
                    new_widgetsIR.append(item)

            wordlist = self.get_screenIRwordlist(screen)
            sent_similarity_matrix = np.zeros((len(widgets), len(new_widgetsIR)))
            word_similarity_matrix = np.zeros((len(widgets), len(new_widgetsIR)))

            for i, v1 in enumerate(sent_text_info):
                if v1 == []:
                    continue
                for j, v2 in enumerate(new_widgetsIR):
                    wordlist1 = self.get_widgetwordlist(v2)
                    score0 = text_sim_bert.calc_similarity(' '.join(v1), wordlist1)
                    sent_similarity_matrix[i][j] = score0


            for i, v1 in enumerate(word_text_info):
                for j, v2 in enumerate(new_widgetsIR):
                    wordlist1 = StrUtil.tokenize('IR', v2, True)  # 列表
                    score1 = self.get_word_simval(v1, wordlist1, text_sim_bert)
                    word_similarity_matrix[i][j] = score1

            similarity_matrix = (sent_similarity_matrix + word_similarity_matrix) / 2
            duplicates = []
            for i in range(len(similarity_matrix[0])):
                col = [row[i] for row in similarity_matrix]
                if col in duplicates:
                    max_index = max(duplicates.index(col), i)
                    for j in range(len(similarity_matrix)):
                        similarity_matrix[j][max_index] = 0  # Change elements of the column with the larger index to 0
                    break
                duplicates.append(col)

            similarity_matrix1 = similarity_matrix

            for i, text in enumerate(sent_text_info):
                if not text:
                    continue
                for j, widgetIR in enumerate(new_widgetsIR):
                    element_type = list(widgets.keys())[i].get_element_type().split('.')[-1]
                    if self.look_for_exact_match(widgetIR, text, usage_name):
                        if widgetIR == "search_bar":
                            if element_type == "EditText" or element_type == "TextView":
                                similarity_matrix1[i][j] = 0.9
                        elif widgetIR == "to_search":
                            if element_type != "EditText":
                                similarity_matrix1[i][j] = 0.9
                        else:
                            similarity_matrix1[i][j] = 0.9

                    if widgetIR in ['menu'] and element_type == "ImageButton" and screen == 'home':
                        if self.convert_bounds_to_wIR(list(widgets.keys())[i].get_middle_point()) == 'menu':
                            similarity_matrix1[i][j] = 0.9
                    if widgetIR in ['search_bar'] and element_type not in ["ImageView"] and screen == 'home':
                        if self.convert_bounds_to_wIR(list(widgets.keys())[i].get_middle_point()) == 'search_bar':
                            similarity_matrix1[i][j] = 0.9

                    if similarity_matrix1[i][j] == 0.9:
                        for k in range(len(new_widgetsIR)):
                            if similarity_matrix1[i][k] != 0.9:
                                similarity_matrix1[i][k] = 0.0
                        for k in range(len(sent_text_info)):
                            if similarity_matrix1[k][j] != 0.9:
                                similarity_matrix1[k][j] = 0.0
                        similarity_matrix1[i][j] = 0.9

            similar_values, pos0 = self.analyze_matrix(similarity_matrix1, widgets, new_widgetsIR, widget_self, zones,
                                                       recynodes00, screen)

            if similar_values.count(1.0) > 1:
                similar_values0 = [1.0] + [x for x in similar_values if x != 1.0]
            else:
                similar_values0 = similar_values

            if similar_values0:
                mean_similar = np.mean(similar_values0)
            else:
                mean_similar = 0


            similarity_ac_IR0 = text_sim_bert.calc_similarity(activity_wordlist, wordlist)
            wordlist = self.get_wordlist(screen)
            similarity_pagetext = text_sim_bert.calc_similarity(text_info_strs, wordlist)
            similarity_ac_IR = max(similarity_ac_IR0, similarity_pagetext)
            inedges_trigger = self.collect_in_edges(digraph, screen, transitions_list)

            if pre_trigger not in ['up', 'down', 'left', 'right',
                                   'initial'] and pre_trigger in inedges_trigger and pre_trigger != "continue#click":
                similarity_trigger = 1

                Com_similarity = (mean_similar + similarity_ac_IR[0][0] + similarity_trigger) / 3
            else:
                Com_similarity = 0.45 * mean_similar + 0.55 * similarity_ac_IR[0][0]

            similardict[screen] = (Com_similarity, similarity_matrix1, similar_values, pos0)
            #  Set the screen similarity to the amount of information contained in the IR (edge relationships), the similarity between the node IR attributes and activities,
            #  and the maximum similarity between the node IR and component attributes.

        return similardict, widgets, value_1

    def get_wordlist(self, screenIR):
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        wordlist_dir = os.path.join(current_dir_path, '..', '..', 'IR', 'label_texts')
        wordlist_path = os.path.join(wordlist_dir, screenIR + '.txt')
        if os.path.exists(wordlist_path):
            file = open(wordlist_path, "r")
            return file.read().lower()

        return screenIR  # if no wordlist is found for the screenIR, use the screenIR name itself as the wordlist

    def process(self, widgets, recynodes0, bottom_bounds, zones):

        resource_ids = {}
        nodes_in_recy = []
        for element in widgets:
            if 1 in zones.keys() and element in zones[1]:
                attrs = widgets[element]
                resource_id = attrs.get('resource-id')
                if resource_id:
                    if resource_id in resource_ids:
                        resource_ids[resource_id].append(element)
                    else:
                        resource_ids[resource_id] = [element]

        len_min = 1
        for resource_id, elements in resource_ids.items():
            if len(elements) > len_min:
                nodes_in_recy = elements
                len_min = len(elements)

        for key, value in widgets.items():
            if key in nodes_in_recy and len(widgets[key]) > 1:
                del widgets[key]['resource-id']

        if nodes_in_recy != [] and nodes_in_recy[0] in zones[1]:
            if len(recynodes0) < 3 and recynodes0 != [] and len(recynodes0[0].attributes['text'].split()) - len(
                    nodes_in_recy[0].attributes['text'].split()) > 10:
                recynodes0 = nodes_in_recy
            else:
                recynodes0.extend(nodes_in_recy)

        if len(recynodes0) != 0:
            total_length = 0
            for i in recynodes0:
                total_length += len(i.attributes['text'].split())

            average_length = total_length / len(recynodes0)

            updated_list = []

            if len(recynodes0) > 5:
                for i in recynodes0:
                    if abs(len(i.attributes['text'].split()) - average_length) < 10:
                        updated_list.append(i)
                recynodes0 = updated_list
        if bottom_bounds != 0:
            for recynode in recynodes0:
                if 3 in zones.keys() and recynode in zones[3] and self.is_inbounds(bottom_bounds,
                                                                                   recynode.get_middle_point()):
                    recynodes0.remove(recynode)
                    zones[3].remove(recynode)

                    key = recynode
                    index = 0
                    for k in widgets.keys():
                        if k == key:
                            del widgets[k]
                            break
                        index += 1
        recynodes0 = list(set(recynodes0))

        resource_ids = {}
        for element in widgets:
            attrs = widgets[element]
            resource_id = attrs.get('resource-id')
            if resource_id:
                if resource_id in resource_ids:
                    resource_ids[resource_id].append(element)
                else:
                    resource_ids[resource_id] = [element]

        for resource_id, elements in resource_ids.items():

            if len(elements) > 1:
                for element in widgets:
                    resource_id0 = widgets[element].get('resource-id')
                    if resource_id0 and resource_id0 == resource_id and len(widgets[element]) != 1:
                        del widgets[element]['resource-id']

        for key, value in widgets.items():
            keys_to_remove = []
            for inner_key, inner_value in value.items():
                if inner_value == '':
                    keys_to_remove.append(inner_key)

            for inner_key in keys_to_remove:
                del widgets[key][inner_key]

        return widgets, recynodes0, zones

    def get_childrens(self, node):
        childrens = node.children
        for children_i in childrens:
            if children_i.children != [] and children_i.get_element_type().split('.')[-1] == "LinearLayout":
                continue
            else:
                childrens.extend(children_i.children)

        return childrens

    def get_childrens_semantic(self, node, childrens):

        d = {'text': ''}
        for ele_i in childrens:
            if ele_i.attributes['clickable'] == "false":
                # print("ele_i", ele_i, vars(ele_i))
                if 'text' in ele_i.attributes.keys() and ele_i.attributes['text'] != "":
                    d, node = self.adjust_node_d(ele_i, d, node)
                else:
                    if ele_i.get_element_type().split('.')[-1] in ["ViewGroup", "LinearLayout", "FrameLayout",
                                                                   "RelativeLayout"]:
                        if len(ele_i.children) == 1 and ele_i.children[0].get_element_type().split('.')[-1] in [
                            "RelativeLayout"]:
                            childrens0 = ele_i.children[0].children
                        else:
                            childrens0 = ele_i.children
                        for ele_i_ch in childrens0:
                            if 'text' in ele_i_ch.attributes.keys() and ele_i_ch.attributes['text'] != "":
                                d, node = self.adjust_node_d(ele_i_ch, d, node)
                            else:
                                if ele_i_ch.get_element_type().split('.')[-1] in ["ViewGroup", "LinearLayout",
                                                                                  "FrameLayout", "RelativeLayout"]:
                                    if len(ele_i_ch.children) == 1 and \
                                            ele_i_ch.children[0].get_element_type().split('.')[-1] in [
                                        "RelativeLayout"]:
                                        childrens00 = ele_i_ch.children[0].children
                                    else:
                                        childrens00 = ele_i_ch.children
                                    for ele_i_ch_ch in childrens00:
                                        if 'text' in ele_i_ch_ch.attributes.keys() and ele_i_ch_ch.attributes[
                                            'text'] != "":
                                            d, node = self.adjust_node_d(ele_i_ch_ch, d, node)

        if d == {'text': ''}:
            d = {}
        return d, node

    def adjust_node_d(self, ele, d, node):
        if len(ele.attributes['text'].split()) > len(d['text'].split()):
            for key in ['text', 'content-desc']:
                d[key] = ele.attributes[key]
                node.exec_identifier[key] = ele.attributes[key]
                node.attributes[key] = ele.attributes[key]
            d['class'] = ele.attributes['class']
            node.attributes['class'] = ele.attributes['class']
            node.exec_identifier['class'] = ele.attributes['class']
            d['resource-id'] = ele.attributes['resource-id']
            node.attributes['bounds'] = ele.attributes['bounds']
        return d, node

    def collect_in_edges(self, digraph, screen, transitions_list):
        predecessor_nodes = list(digraph.predecessors(screen))
        if screen in predecessor_nodes:
            predecessor_nodes.remove(screen)
        from_trigger = []
        for source in predecessor_nodes:
            for item in transitions_list:
                if source == item['source'] and screen == item['dest']:
                    from_trigger.append(item['trigger'])
        return from_trigger

    def look_for_exact_match(self, trigger, text, usage_name):
        if len(text) > 18:
            return False
        if "_" in trigger:
            v1 = trigger.replace("_", "")
            if v1 in text:
                return True
            v2 = trigger.replace("_", " ")
            if v2 in text:
                return True
        if "_" in trigger:
            trigger_words = trigger.split("_")
            if len(trigger_words) > 2 and len(text) == 1:
                return False
            for word in trigger_words:
                if word in ["by", "i", "multi", "to", "sign", "in", "up", "or", "form", "settings"]:
                    continue
                if word in text:
                    return True
        if trigger == "to_signin_or_signup":
            usage = [usage_name.split('-')[-1].lower()]
            usage = StrUtil.split_text(usage)

            if ("sign" in text) and ("in" in text) or ("signin" in text) or ("login" in text) and ("in" in text):
                return True
            if ("sign" in text) and ("up" in text) or ("signup" in text) or ("register" in text) or (
                    "create" in text and "account" in text):
                return True
            if ("google" in text) or ("facebook" in text):
                return False
        if trigger == "sign_up":
            if ("register" in text) or ("create" in text and "account" in text) or ("join" in text) or (
                    "get" in text and "started" in text):
                return True
        if trigger == "sign_in":
            if ("get" in text and "started" in text):
                return True
        if trigger == "by_google":
            if "google" in text:
                return True
        if trigger == "email":
            if ("identifier" in text) or ("phone" in text):
                return True
        if trigger == "continue":
            if ("ok" in text) or ("accept" in text) or ("next" in text) or ("get" in text and "started" in text) or (
                    "take" in text and "to" in text) or "allow" in text:
                if len(text) < 10:
                    return True
        if trigger == "category":
            if ("sections" in text) or ("topics" in text) or ("navigation" in text) or ("menu" in text) or (
                    "browse" in text):
                return True
        if trigger == "bypass":
            if ("deny" in text) or ("skip" in text):
                return True
        if trigger == "menu":
            if ("drawer" in text) or ("navigation" in text) or ("option" in text) or ("options" in text) or (
                    "browse" in text) or ("navigate" in text) and ("up" in text) or ("settings" in text) or (
                    "open" in text) and ("menu" in text):
                return True
        if trigger == "menu_settings":
            if "settings" in text and len(text) < 5:
                return True
        if trigger == "menu_more":
            if "more" in text:
                return True

        if trigger == "search_bar":
            if "search" in text:
                return True

        if trigger == "to_search":
            if "search" in text:
                return True
        if trigger == "cart":
            if "bag" in text or "badge" in text:
                return True
        if trigger == "buy":
            if "continue" in text:
                return True
        if trigger == "help":
            if ("guide" in text) or ("question" in text) or ("faq" in text) or ("how" in text) or ("questions" in text):
                return True
        if "save" in trigger:
            if "save" in text:
                return True

        if "bookmark" in trigger:
            if "bookmark" in text or "save" in text:
                return True
        if "apply" in trigger or trigger == "apply":
            if "done" in text or "submit" in text or "send" in text or "agree" in text or "ok" in text or "allow" in text:
                return True
        if "item" in trigger:
            if "product" in text:
                return True
        if "back" in trigger:
            if "navigate" in text:
                return True
        if "account" in trigger:
            if "you" in text:
                return True
            if "profile" in text:
                return True
        if "item_option" in trigger:
            if ("10pcs") in text or ("20pcs") in text or ("50pcs") in text:
                return True

        if "multi_form" in trigger:
            if "create" and "account" in text and len(text) == 2:
                return True

        IRt = StrUtil.tokenize('IR', trigger, False)
        if len(IRt) > 2 and len(set(text) & set(IRt)) > len(IRt) // 2 or len(set(text) & set(IRt)) == len(IRt):
            if trigger == "email":
                if "forgot" in text:
                    return False
            return True
        if trigger in text:
            return True
        return False

    def check_for_top_match_heuristics(self, widgetIR, screenIR, element):
        if screenIR == "menu":
            if element.is_a_list_item():
                return True


if __name__ == '__main__':
    state = State('')
    state.UIXML_path = 'E:/2/GUEST/GUEST/Final-Artifacts/output/models/18-Textsize/dynamic_output/fox/screenshots/0-1.xml'
    # state.nodes = ['a', 'b']
    print('all done! :)')
