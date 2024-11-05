import xml.etree.ElementTree as ET
from queue import Queue
from PIL import Image
from io import BytesIO
import base64
import os
from lxml import etree
import re
import node
import state


class LayoutTree:
    statusbar_h = 0
    navbar_h = 0

    def __init__(self, driver, pathName):
        self.pagesrc = driver.page_source
        pattern = r"&#\d+;"
        invalid_entities = re.findall(pattern, self.pagesrc)
        for entity in invalid_entities:
            self.pagesrc = self.pagesrc.replace(entity, '')
        self.root = ET.fromstring(self.pagesrc)
        self.pathName = pathName
        info = driver.get_system_bars()
        self.statusbar_h = info["statusBar"]["height"]
        self.navbar_h = info["navigationBar"]["height"]
        try:
            import os
            os.mkdir("{}/pngs".format(pathName))
        except:
            pass
        data = base64.decodebytes(driver.get_screenshot_as_base64().encode())
        self.screenshot = Image.open(BytesIO(data))
        self.screenshotWidth, self.screenshotHeight = self.screenshot.size

    def createDirs(self):
        from shutil import rmtree
        path = "./{}/{}/pngs".format(self.appName, self.stateName)
        try:
            rmtree(path)
        except:
            print("Failed to remove tree. Prob not existing")
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

    def androidBoundtransform(self, boundStr):
        if boundStr is not None:
            pattern = re.compile(r"\[(?P<x1>[\d]+),(?P<y1>[\d]+)\]\[(?P<x2>[\d]+),(?P<y2>[\d]+)\]")
            match = pattern.match(boundStr)
            if match:
                x = int(match.group('x1'))
                y = int(match.group('y1'))
                x2 = int(match.group('x2'))
                y2 = int(match.group('y2'))
                width = x2 - x
                height = y2 - y
        return x, y, width, height

    def saveCroppedImageV2(self, x, y, width, height):
        actualX = int(x)
        actualY = int(y)
        actualWidth = int(width)
        actualHeight = int(height)

        x1 = actualX
        y1 = actualY
        x2 = actualX + actualWidth
        y2 = actualY + actualHeight

        if y2 == self.scaleHeigh:
            y2 += self.statusbar_h
        b = (x1, y1, x2, y2)
        cropSuffix = str(x1) + "_" + str(y1) + "_" + str(x2) + "_" + str(
            actualY + actualHeight)  # fix for 2200(relative)/2274(true) to match the xml bound
        crop = self.screenshot.crop(box=b)
        name = "{}/pngs/{}.png".format(self.pathName, cropSuffix)
        crop.save(name)
        crop = crop.convert('L')
        bounds = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        return name, bounds

    def isElementInside(self, ele_x, ele_y, node):
        if ele_x is None or ele_y is None or node is None:
            return False
        if "width" in node.attrib:
            if (int(node.attrib["x"]) <= ele_x < int(node.attrib["x"]) + int(node.attrib["width"])) and (
                    int(node.attrib["y"]) <= ele_y < int(node.attrib["y"]) + int(node.attrib["height"])):
                return True, int(node.attrib["width"]) * int(node.attrib["height"])
        elif "bounds" in node.attrib:
            boundStr = node.attrib['bounds']
            if boundStr != None:
                x, y, width, height = self.androidBoundtransform(boundStr)
                if (x <= ele_x < x + width) and (y <= ele_y < y + height):
                    return True, width * height
            else:
                print(str.format("Node: {} has no boundary, skipped", node.tag))
        return False, None

    def isElementMatching(self, saveElementBoundsRect, node):
        if saveElementBoundsRect == None:
            return False

        if "width" in node.attrib:
            if (saveElementBoundsRect["width"] == int(node.attrib["width"]) and saveElementBoundsRect["height"] == int(
                    node.attrib["height"]) and saveElementBoundsRect["x"] == int(node.attrib["x"]) and
                    saveElementBoundsRect["y"] == int(node.attrib["y"])):
                return True
        elif "bounds" in node.attrib:
            if node.attrib["bounds"] != None:
                x, y, width, height = self.androidBoundtransform(node.attrib["bounds"])
                if (saveElementBoundsRect["width"] == width and saveElementBoundsRect["height"] == height and
                        saveElementBoundsRect["x"] == int(x) and saveElementBoundsRect["y"] == int(y)):
                    return True
            else:
                print(str.format("Node: {} has no boundary, skipped", node.tag))
        return False

    def findElementByImagePath(self, imagePath):
        x1 = x2 = y1 = y2 = 0
        filename = os.path.basename(imagePath)
        boundRegex = re.compile(r"(\d+)_(\d+)_(\d+)_(\d+)")
        match = boundRegex.match(filename)
        if match:
            x1 = int(match.group(1))
            y1 = int(match.group(2))
            x2 = int(match.group(3))
            y2 = int(match.group(4))

        utf8_parser = etree.XMLParser(encoding='utf-8')
        root = etree.fromstring(self.pagesrc.encode('utf-8'), parser=utf8_parser)
        tree = etree.ElementTree(root)
        dic = {}
        bound = {}
        for child in root.iter():

            if child.tag == 'XCUIElementTypeApplication':
                self.scaleWidth = float(child.attrib["width"])
                self.scaleHeigh = float(child.attrib["height"])
                print(self.scaleWidth, self.scaleHeigh)
            if not 'displayed' in child.attrib:
                continue
            numOfChildren = len(child)
            if child.attrib['displayed'] == 'true':
                bound["width"] = x2 - x1
                bound["height"] = y2 - y1
                bound["x"] = x1
                bound["y"] = y1

                if self.isElementMatching(bound, child):
                    print("The matched element is:" + tree.getpath(child))
                    dic[child] = numOfChildren

        temp = min(dic.values())
        res = [key for key in dic if dic[key] == temp]
        minElement = res[0]
        print("The mininum matched child is:" + tree.getpath(minElement))

        return minElement

    def findMinElement(self, ele_x, ele_y):
        utf8_parser = etree.XMLParser(encoding='utf-8')
        root = etree.fromstring(self.pagesrc.encode('utf-8'), parser=utf8_parser)
        tree = etree.ElementTree(root)
        dic = {}
        minElement = None
        for child in root.iter():
            if child.tag == 'XCUIElementTypeApplication':
                self.scaleWidth = float(child.attrib["width"])
                self.scaleHeigh = float(child.attrib["height"])
                print(self.scaleWidth, self.scaleHeigh)
            if not 'displayed' in child.attrib:
                continue
            numOfChildren = len(child)
            if child.attrib['displayed'] == 'true':
                isinside, size = self.isElementInside(ele_x, ele_y, child)
                if isinside:
                    dic[child] = size
        temp = min(dic.values())
        res = [key for key in dic if dic[key] == temp]
        minElement = res[0]
        print("The enclosing element is:")
        print(tree.getpath(minElement))
        if "bounds" in minElement.attrib:
            if minElement.attrib["bounds"] != None:
                x, y, width, height = self.androidBoundtransform(minElement.attrib["bounds"])
                screenshotFileName, tmpBounds = self.saveCroppedImageV2(x, y, width, height)
        return screenshotFileName, tmpBounds, ""

    def extract_state(self, saveElementBoundsRect=None, saveCrops=True, onlyVisible=False):
        counter = 0
        f = open("{}/graph.dot".format(self.pathName), "w")
        f.write("digraph Layout {\n\n\tnode [shape=record fontname=Arial];\n\n")
        curr_state = state.State(self.screenshot)
        queue = Queue()
        queue.put(self.root)
        parent = {}
        numInParent = {}
        numbering = {}
        ordering = ""
        pngToMatchedElementPath = None
        bounds = None

        while not queue.empty():
            source_node = queue.get()
            if source_node.tag == 'hierarchy':
                self.scaleWidth = 1080
                self.scaleHeigh = 1794

            num_of_children = len(list(source_node))

            info = "{"
            for key, value in source_node.attrib.items():
                if key in ["index", "package"]:
                    continue
                if key != "text":
                    info += "|"
                info += key + " = " + value + "\\l"

            info += "|numberOfChildren = " + str(num_of_children) + "\\l"
            if source_node in numInParent:
                info += "|numInParentLayout = " + str(numInParent[source_node]) + "\\l"

            isElementMatched = self.isElementMatching(saveElementBoundsRect, source_node)
            if isElementMatched:
                info += "|eventGeneratedOnElement = true \\l"
            else:
                info += "|eventGeneratedOnElement = false \\l"

            is_interactable = False
            interactions = []
            for key, value in source_node.attrib.items():
                # print("source_node.attrib.items()",source_node.attrib.items())
                if key == "checkable" and value == "true":
                    is_interactable = True
                    interactions.append("check")
                if key == "long-clickable" and value == "true":
                    is_interactable = True
                    interactions.append("long-click")
                if key == "clickable" and value == "true":
                    is_interactable = True
                    interactions.append("click")
                if key == "focusable" and value == "true":
                    is_interactable = True
                    interactions.append("focus")

            screenshotFileName = ""
            if saveCrops:  # and node.attrib["displayed"] == "true"
                if is_interactable:
                    # print("Saving crop!")
                    if 'bounds' in source_node.attrib:
                        if source_node.attrib["bounds"] != None:
                            x, y, width, height = self.androidBoundtransform(source_node.attrib["bounds"])
                            if not (width == 0 and height == 0):
                                screenshotFileName, tmpBounds = self.saveCroppedImageV2(x, y, width, height)
                                info += "|screenshotPath = " + screenshotFileName + "\\l"
                                if isElementMatched:
                                    pngToMatchedElementPath = screenshotFileName
                                    bounds = tmpBounds
                            else:
                                print("[ERROR] Skipping crop on 0x0 node." + source_node.tag)
                        else:
                            print(str.format("Node: {} has no boundary, skipped", source_node.tag))

            curr_node = node.Node(counter, source_node.attrib, is_interactable, interactions, num_of_children,
                                  source_node.tag, screenshotFileName)
            info += "}"
            string = "\t" + str(counter) + "\t[label=\"" + info + "\"]\n"
            numbering[source_node] = counter
            f = open("{}/graph.dot".format(self.pathName), "a", encoding='utf-8')
            f.write(string)
            if source_node in parent:
                parent_node = curr_state.get_node(numbering[parent[source_node]])
                curr_node.parent = parent_node
                parent_node.add_child(curr_node)
                ordering += "\t" + str(numbering[parent[source_node]]) + " -> " + str(counter) + "\n"

            curr_state.add_node(curr_node)

            counter = counter + 1
            parent_counter = {}
            for child in source_node:
                parent[child] = source_node
                queue.put(child)
                child_class = child.attrib["class"]
                if child_class not in parent_counter:
                    parent_counter[child_class] = 0
                numInParent[child] = parent_counter[child_class]
                parent_counter[child_class] += 1

        f.write("\n\n")
        f.write(ordering)
        f.write("\n\n}")
        f.close

        if (not pngToMatchedElementPath or not bounds) and saveElementBoundsRect:
            # print("The passed element is not found in the layout tree.\n{}".format(saveElementBoundsRect))
            # print("[INFO] Failsafe. Using FindMinElement")
            pngToMatchedElementPath, bounds = self.findMinElement(saveElementBoundsRect["x"],
                                                                  saveElementBoundsRect["y"])
            print(pngToMatchedElementPath)

        return curr_state

    def printTree(self, saveElementBoundsRect=None, saveCrops=True, onlyVisible=False):
        counter = 0
        f = open("{}/graph.dot".format(self.pathName), "w")
        textual_data_file = open("{}/texts.txt".format(self.pathName), "a")
        f.write("digraph Layout {\n\n\tnode [shape=record fontname=Arial];\n\n")
        queue = Queue()
        queue.put(self.root)
        parent = {}
        numbering = {}
        numInParent = {}
        ordering = ""

        pngToMatchedElementPath = None
        bounds = None
        textual_info = ""
        while not queue.empty():
            element_text = ""
            node = queue.get()
            has_name = False
            print("node.tag", node.tag)

            if node.tag == 'hierarchy':
                self.scaleWidth = int(node.attrib["width"])
                self.scaleHeigh = int(node.attrib["height"])
                print(self.scaleWidth, self.scaleHeigh)

            numOfChildren = len(list(node))

            info = "{"
            for key, value in node.attrib.items():
                if key in ["index", "package"]:
                    continue
                if key != "class":
                    info += "|"
                info += key + " = " + value + "\\l"

            info += "|numberOfChildren = " + str(numOfChildren) + "\\l"
            if node in numInParent:
                info += "|numInParentLayout = " + str(numInParent[node]) + "\\l"

            isElementMatched = self.isElementMatching(saveElementBoundsRect, node)
            if isElementMatched:
                info += "|eventGeneratedOnElement = true \\l"
            else:
                info += "|eventGeneratedOnElement = false \\l"

            is_interactable = False
            for key, value in node.attrib.items():
                if key == "checkable" and value == "true":
                    is_interactable = True
                if key == "long-clickable" and value == "true":
                    is_interactable = True
                # if key == "scrollable" and value == "true":
                #     is_interactable = True
                if key == "clickable" and value == "true":
                    is_interactable = True
                if key == "focusable" and value == "true":
                    is_interactable = True

                if key == "content-desc":
                    element_text += "\"content-desc\" : \"" + value + "\" , "
                    has_name = True
                if key == "id":
                    element_text += "\"id\" : \"" + value + "\" , "
                    has_name = True
                if key == "resource-id":
                    element_text += "\"resource-id\" : \"" + value + "\" , "
                    has_name = True
                if key == "text":
                    element_text += "\"text\" : \"" + value + "\" , "

                if isElementMatched:
                    print(key)
                    if key == "content-desc":
                        textual_info += "\"content-desc\" : \"" + value + "\" , "
                    if key == "id":
                        textual_info += "\"id\" : \"" + value + "\" , "
                    if key == "resource-id":
                        textual_info += "\"resource-id\" : \"" + value + "\" , "
                    if key == "text":
                        textual_info += "\"text\" : \"" + value + "\" , "

            if saveCrops:  # and node.attrib["displayed"] == "true"
                if is_interactable:
                    print("Saving crop!")
                    if 'bounds' in node.attrib:
                        if node.attrib["bounds"] != None:
                            x, y, width, height = self.androidBoundtransform(node.attrib["bounds"])
                            if not (width == 0 and height == 0):
                                screenshotFileName, tmpBounds = self.saveCroppedImageV2(x, y, width, height)
                                if has_name:
                                    textual_data_file.write("{ " + element_text)
                                    textual_data_file.write("\"screenshotPath\" : \"" + screenshotFileName + "\"}\n")
                                info += "|screenshotPath = " + screenshotFileName + "\\l"
                                if isElementMatched:
                                    pngToMatchedElementPath = screenshotFileName
                                    bounds = tmpBounds
                            else:
                                print("[ERROR] Skipping crop on 0x0 node." + node.tag)
                        else:
                            print(str.format("Node: {} has no boundary, skipped", node.tag))
            info += "}"
            # print(info)
            string = "\t" + str(counter) + "\t[label=\"" + info + "\"]\n"
            numbering[node] = counter
            f.write(string)
            if node in parent:
                ordering += "\t" + str(numbering[parent[node]]) + " -> " + str(counter) + "\n"
            counter += 1
            i = 0
            for child in node:
                parent[child] = node
                if not onlyVisible:  # or child.attrib["displayed"] == "true"
                    queue.put(child)
                    numInParent[child] = i
                i += 1
        f.write("\n\n")
        f.write(ordering)
        f.write("\n\n}")
        f.close

        # failsafe
        if (not pngToMatchedElementPath or not bounds) and saveElementBoundsRect:
            print("The passed element is not found in the layout tree.\n{}".format(saveElementBoundsRect))
            print("[INFO] Failsafe. Using FindMinElement")
            falsify_text = ""
            pngToMatchedElementPath, bounds, falsify_text = self.findMinElement(saveElementBoundsRect["x"],
                                                                                saveElementBoundsRect["y"])
            print(pngToMatchedElementPath)

        if textual_info != "":
            textual_info = textual_info[:-3]
        return pngToMatchedElementPath, bounds, textual_info

    def saveScreenshot(self):
        self.screenshot.save("{}_whole.png".format(self.name))


if __name__ == "__main__":
    import shutil

    path = "{}/{}/pngs".format("APP1", "S0")
    try:
        shutil.rmtree(path)
    except:
        print("Failed to removed dir. Not existing.")

    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
