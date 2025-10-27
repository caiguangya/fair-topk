/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
 
#ifndef FAIR_TOPK_TOURNEYTREE_H
#define FAIR_TOPK_TOURNEYTREE_H

#include <algorithm>
#include <vector>
#include <iterator>
#include <limits>
#include <cmath>
#include <concepts>
#include <cassert>
#include <type_traits>
#include <cstring>
#include <utility>

#include "memory.h"

namespace FairTopK {

template <class Compare, class CrossCompute, class Line>
concept LegitTourneyLineTree = requires(Compare compare, CrossCompute crossCompute, Line line, double t) {
    { compare(line, line, t) } -> std::convertible_to<bool>;
    { crossCompute(line, line) } -> std::same_as<double>;
};

template <class EquivCompare, class Func, class Line>
concept LegitApplyToEquivs = requires(EquivCompare equivCompare, Func func, Line line) {
    { equivCompare(line, line) } -> std::convertible_to<bool>;
    { func(line) } -> std::same_as<void>;
};

namespace Detail {

enum class EventType : unsigned int { Top = 1, Left = 2, Right = 3 };
enum class TopType : unsigned int { Left = 4, Right = 8 };

struct OptimizedNode {
    OptimizedNode *getLeft() const noexcept { return (OptimizedNode *)this + 1; /* Pre-order traversal layout */ }
    OptimizedNode *getRight() const noexcept { return rightChild; }
    void setSecondChild(OptimizedNode *right) noexcept {
        rightChild = right;
    }
    EventType getEventType() const noexcept {
        return EventType(flags & eventMask);
    }
    void setEventType(EventType type) noexcept {
        flags &= (~eventMask); 
        flags |= (unsigned int)type;
    }
    TopType getTopType() const noexcept {
        return TopType(flags & topMask);
    }
    void setTopType(TopType type) noexcept {
        flags &= (~topMask);
        flags |= (unsigned int)type;
    }
    void updateNextEvent() noexcept {
        assert(!isLeaf());

        nextEventTime = topChangeTime;
        EventType type = EventType::Top;
        OptimizedNode *left = getLeft();
        OptimizedNode *right = getRight();

        if (left->nextEventTime < nextEventTime) {
            nextEventTime = left->nextEventTime;
            type = EventType::Left;
        }
        if (right->nextEventTime < nextEventTime) {
            nextEventTime = right->nextEventTime;
            type = EventType::Right;
        }
        setEventType(type);
    }
    bool isLeaf() const noexcept { return rightChild == nullptr; /* Full tree */ }

    int topLineIdx = -1;
private:
    unsigned int flags = 0;
    OptimizedNode *rightChild = nullptr;
public:
    double topChangeTime;
    double nextEventTime;
private:
    static constexpr unsigned int eventMask = 0x3;
    static constexpr unsigned int topMask = 0xC;
};

struct UnoptimizedNode {
    UnoptimizedNode *getLeft() const noexcept { return leftChild; }
    UnoptimizedNode *getRight() const noexcept { return rightChild; }
    void setLeftChild(UnoptimizedNode *left) noexcept {
        leftChild = left;
    }
    void setRightChild(UnoptimizedNode *right) noexcept {
        rightChild = right;
    }
    EventType getEventType() const noexcept {
        return eventType;
    }
    void setEventType(EventType type) noexcept {
        eventType = type;
    }
    TopType getTopType() const noexcept {
        return topType;
    }
    void setTopType(TopType type) noexcept {
        topType = type;
    }
    void updateNextEvent() noexcept {
        assert(!isLeaf());

        nextEventTime = topChangeTime;
        EventType type = EventType::Top;
        UnoptimizedNode *left = getLeft();
        UnoptimizedNode *right = getRight();

        if (left->nextEventTime < nextEventTime) {
            nextEventTime = left->nextEventTime;
            type = EventType::Left;
        }
        if (right->nextEventTime < nextEventTime) {
            nextEventTime = right->nextEventTime;
            type = EventType::Right;
        }
        setEventType(type);
    }
    bool isLeaf() const noexcept { return leftChild == nullptr && rightChild == nullptr; }

    int topLineIdx = -1;
    double topChangeTime;
    double nextEventTime;
private:
    UnoptimizedNode *leftChild = nullptr;
    UnoptimizedNode *rightChild = nullptr;
    EventType eventType;
    TopType topType;
};

void recursiveDestory(UnoptimizedNode *node) {
    if (node == nullptr) return;
        
    if (!node->isLeaf()) {
        recursiveDestory(node->getLeft());
        recursiveDestory(node->getRight());
    }

    delete node;
}

}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
class KineticTourneyLineTree {
public:
    KineticTourneyLineTree(std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, 
        double beginTime, double endTime, Compare compare, CrossCompute crossTimeCompute);
    bool replaceTop(const Line& line);
    const Line& Top() const { return lines[root->topLineIdx]; }
    double getNextEventTime() const {  return root->nextEventTime; }

    bool Advance();
    template<class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
    void applyToTopEquivs(EquivCompare equivCompare, Func func) const;

    ~KineticTourneyLineTree();
    KineticTourneyLineTree(const KineticTourneyLineTree&) = delete;
    KineticTourneyLineTree(KineticTourneyLineTree&&) = delete;
    KineticTourneyLineTree& operator=(const KineticTourneyLineTree&) = delete;
    KineticTourneyLineTree& operator=(KineticTourneyLineTree&&) = delete;
private:
    using Node = std::conditional<Opt, Detail::OptimizedNode, Detail::UnoptimizedNode>::type;

    Node *recursiveBuild(std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, int& idx);
    void recursiveReplaceTop(const Line& line, Node *node);
    bool recursiveAdvance(Node *node);

    void updateNodeTop(int leftTopLineIdx, int rightTopLineIdx, Node *node);
    template<class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
    void applyToTopEquivs(const Line& topLine, const Node *node, EquivCompare&& equivCompare, Func&& func) const;

    Compare compare;
    CrossCompute crossTimeCompute;
    double curTime = 0.0;
    double endTime = 0.0;

    std::vector<Line> lines;
    Node *root = nullptr;
};

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
inline void KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::updateNodeTop(int leftTopLineIdx, int rightTopLineIdx, 
    Node *node) {
    const Line &topLeft = lines[leftTopLineIdx];
    const Line &topRight = lines[rightTopLineIdx];

    bool futureExchange = false;
    if (compare(topLeft, topRight, curTime)) {
        node->topLineIdx = leftTopLineIdx;
        node->setTopType(Detail::TopType::Left);
        futureExchange = compare(topRight, topLeft, endTime);
    }
    else {
        node->topLineIdx = rightTopLineIdx;
        node->setTopType(Detail::TopType::Right);
        futureExchange = compare(topLeft, topRight, endTime);
    }

    if (futureExchange) {
        node->topChangeTime = std::clamp(crossTimeCompute(topLeft, topRight), curTime, endTime);
    }
    else {
        node->topChangeTime = std::numeric_limits<double>::infinity();
    }
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::KineticTourneyLineTree(std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, 
    double beginTime, double endTime, Compare compare, CrossCompute crossTimeCompute) : 
    curTime(beginTime), endTime(endTime), compare(std::move(compare)), crossTimeCompute(std::move(crossTimeCompute)) {

    auto lineCount = std::distance(begin, end);
    lines.reserve(lineCount);
    std::copy(begin, end, std::back_inserter(lines));

    int idx = 0;
    if constexpr (Opt) {
        root = FairTopK::allocAligned<Node>(2 * lineCount - 1);
        recursiveBuild(lines.cbegin(), lines.cend(), idx);
    }
    else {
        root = recursiveBuild(lines.cbegin(), lines.cend(), idx);
    }
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::~KineticTourneyLineTree() {
    if constexpr (Opt) FairTopK::freeAligned(root);
    else Detail::recursiveDestory(root);
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::Node *KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::recursiveBuild(
    std::vector<Line>::const_iterator begin, std::vector<Line>::const_iterator end, int& idx) {
    if (begin == end) return nullptr;

    Node *node = nullptr;
    if constexpr (Opt) node = new (&root[idx++]) Node();
    else node = new Node(); 

    auto dis = std::distance(begin, end);
    if (dis <= 1) {
        node->topLineIdx = std::distance(lines.cbegin(), begin);
        node->topChangeTime = std::numeric_limits<double>::infinity();
        node->nextEventTime = std::numeric_limits<double>::infinity();
        return node;
    }

    auto mid = begin + dis / 2;
    Node *left = recursiveBuild(begin, mid, idx);
    Node *right = recursiveBuild(mid, end, idx);

    if constexpr (Opt) {
        node->setSecondChild(right);
    }
    else {
        node->setLeftChild(left);
        node->setRightChild(right);
    }
    
    updateNodeTop(left->topLineIdx, right->topLineIdx, node);

    node->updateNextEvent();

    return node;
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
bool KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::replaceTop(const Line& line) {
    int curTopLineIdx = root->topLineIdx;
    lines[curTopLineIdx] = line;
    if (!root->isLeaf())  [[likely]]
        recursiveReplaceTop(line, root);

    return root->topLineIdx != curTopLineIdx;
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
void KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::recursiveReplaceTop(const Line& line, Node *node) {
    assert(!node->isLeaf());

    Node *left = node->getLeft();
    Node *right = node->getRight();

    Detail::TopType topType = node->getTopType();

    if (topType == Detail::TopType::Left) {
        if (!left->isLeaf()) recursiveReplaceTop(line, left);
    }
    else {
        if (!right->isLeaf()) recursiveReplaceTop(line, right);
    }

    if constexpr (!Opt) { // Simulate node insertion and deletion
        bool isLeft = (topType == Detail::TopType::Left);
        Node *topChild = isLeft ? left : right;
        if (topChild->isLeaf()) {
            Node *newChild = new Node();
            std::memcpy(newChild, topChild, sizeof(Node));

            if (isLeft) { 
                node->setLeftChild(newChild);
                left = newChild;
            }
            else { 
                node->setRightChild(newChild);
                right = newChild;
            }

            delete topChild;
        }
    }

    updateNodeTop(left->topLineIdx, right->topLineIdx, node);

    node->updateNextEvent();
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
bool KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::Advance() {
    curTime = root->nextEventTime;

    if (root->isLeaf()) [[unlikely]]
        return false;

    return recursiveAdvance(root);
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
bool KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::recursiveAdvance(Node *node) {
    assert(!node->isLeaf());

    bool topUpdated = false;
    Node *left = node->getLeft();
    Node *right = node->getRight();
    Detail::EventType eventType = node->getEventType();
    if (eventType == Detail::EventType::Top) {
        bool isLeft = (node->getTopType() == Detail::TopType::Left);

        Detail::TopType newTopType = isLeft ? Detail::TopType::Right : Detail::TopType::Left;
        node->topLineIdx = isLeft ?  right->topLineIdx : left->topLineIdx;
        node->topChangeTime = std::numeric_limits<double>::infinity();
        node->setTopType(newTopType);

        node->updateNextEvent();
        topUpdated = true;
    }
    else {
        bool childTopUpdated = false;
        if (eventType == Detail::EventType::Left) {
            if (!left->isLeaf()) childTopUpdated = recursiveAdvance(left);
        }
        else {
            if (!right->isLeaf()) childTopUpdated = recursiveAdvance(right);
        }

        if (childTopUpdated) {
            int preTopLineIdx = node->topLineIdx;

            updateNodeTop(left->topLineIdx, right->topLineIdx, node);

            if (node->topLineIdx != preTopLineIdx) topUpdated = true;
        }

        node->updateNextEvent();
    }

    return topUpdated;
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
template <class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
void KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::applyToTopEquivs(EquivCompare equivCompare, Func func) const {
    const Line &topLine = lines[root->topLineIdx];
    func(topLine);

    if (!root->isLeaf()) [[likely]]
        applyToTopEquivs(topLine, root, std::move(equivCompare), std::move(func));
}

template <class Line, bool Opt, class Compare, class CrossCompute> requires LegitTourneyLineTree<Compare, CrossCompute, Line>
template <class EquivCompare, class Func> requires LegitApplyToEquivs<EquivCompare, Func, Line>
void KineticTourneyLineTree<Line, Opt, Compare, CrossCompute>::applyToTopEquivs(const Line& topLine, const Node *node, 
    EquivCompare&& equivCompare, Func&& func) const {
    assert(!node->isLeaf());

    const Node *curNode = node;
    do {
        const Node *left = curNode->getLeft();
        const Node *right = curNode->getRight();

        if (curNode->getTopType() == Detail::TopType::Left) {
            const Line& rightLine = lines[right->topLineIdx];
            if (std::forward<EquivCompare>(equivCompare)(topLine, rightLine)) {
                std::forward<Func>(func)(rightLine);
                if (!right->isLeaf())
                    applyToTopEquivs(topLine, right, std::forward<EquivCompare>(equivCompare), std::forward<Func>(func));
            }
            curNode = left;
        }
        else {
            const Line& leftLine = lines[left->topLineIdx];
            if (std::forward<EquivCompare>(equivCompare)(topLine, leftLine)) {
                std::forward<Func>(func)(leftLine);
                if (!left->isLeaf()) 
                    applyToTopEquivs(topLine, left, std::forward<EquivCompare>(equivCompare), std::forward<Func>(func));
            }
            curNode = right;
        }
    } while (!curNode->isLeaf());
}

}

#endif
